"""
CF-NeRF Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from cf_nerf.cf_nerf_datamanager import CFNeRFDataManagerConfig
# from cf_nerf.cf_nerf_model import TemplateModel, TemplateModelConfig
# from cf_nerf.template_nerf_model import TemplateModel, TemplateModelConfig
from cf_nerf.cf_nerf_model import CFNeRFModel, CFNerfModelConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import profiler


@dataclass
class CFNeRFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: CFNeRFPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = CFNeRFDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = CFNerfModelConfig()
    """specifies the model config"""
    linear_probe: bool = False
    """Run linear probe to evaluate instance features. GT masks have to be parsed!"""
    linear_probe_gt_path: Union[str, None] = None
    """Path to gt instance mask (Check for correct resolution!)"""
    linear_probe_num_fruits: int = -1
    """Number of predicted classes/instances"""


class CFNeRFPipeline(VanillaPipeline):
    """Template Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
            self,
            config: CFNeRFPipelineConfig,
            device: str,
            test_mode: Literal["test", "val", "inference"] = "val",
            world_size: int = 1,
            local_rank: int = 0,
            grad_scaler: Optional[GradScaler] = None,
            cluster_object=None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            linear_probe=self.config.linear_probe,
            linear_probe_gt_path=self.config.linear_probe_gt_path
        )
        self.datamanager.to(device)

        self.datamanager.linear_probe = self.config.linear_probe
        self.datamanager.linear_probe_gt_path = self.config.linear_probe_gt_path

        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        if isinstance(self.datamanager.train_pixel_sampler, type(None)):
            self.semantic_pixel_sampler_pairs = None
        else:
            self.semantic_pixel_sampler_pairs = self.datamanager.train_semantic_pixel_sampler.config.pixel_per_pair

        if self.test_mode == "test":
            cluster_file = self.config.datamanager.data

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            test_mode=test_mode,
            grad_scaler=grad_scaler,
            pixel_per_pair=self.semantic_pixel_sampler_pairs,
            cluster_object=cluster_object,
            linear_probe=[self.config.linear_probe, self.config.linear_probe_num_fruits]
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                CFNeRFModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """

        instance_step = self.config.model.training_steps["instance"]

        first_statement = (self.config.model.training_steps["cascaded_freezing"] and
                           instance_step - 1 < step < instance_step + 2)

        second_statement = (self.trainer_ignore_instance_weights and
                            (self.set_continue_step + 1 - self.config.model.training_steps["instance"] == 0))

        if first_statement or second_statement:

            for param in self.model.fruit_field.mlp_base_mlp.parameters():
                param.requires_grad = False

            for param in self.model.fruit_field.mlp_base_grid.parameters():
                param.requires_grad = False

            for param in self.model.fruit_field.mlp_base.parameters():
                param.requires_grad = False

            for param in self.model.fruit_field.mlp_head.parameters():
                param.requires_grad = False

            for param in self.model.fruit_field.mlp_semantics.parameters():
                param.requires_grad = False

            for param in self.model.fruit_field.field_head_semantics.parameters():
                param.requires_grad = False

            for param in self.model.fruit_field.position_encoding.parameters():
                param.requires_grad = False

            for param in self.model.fruit_field.spatial_distortion.parameters():
                param.requires_grad = False

            for param in self.model.fruit_field.embedding_appearance.parameters():
                param.requires_grad = False

            for param in self.model.fruit_field.params['semantics'].parameters():
                param.requires_grad = False

            for param in self.model.fruit_field.params['base'].parameters():
                param.requires_grad = False

        ray_bundle, batch = self.datamanager.next_train(step=step, training_steps=self.config.model.training_steps)
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict, step)

        return model_outputs, loss_dict, metrics_dict

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module."):] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: Optional[bool] = None):
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # if "mlp_instance" in key:
                #    continue
                # elif "field_head_instance" in key:
                #    continue
                # elif "params.instance" in key:
                #    continue

                # remove the "_model." prefix from key
                model_state[key[len("_model."):]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module."):]: value for key, value in model_state.items()}

        pipeline_state = {key: value for key, value in state_dict.items() if not key.startswith("_model.")}

        try:
            self.model.load_state_dict(model_state, strict=True)
        except RuntimeError:
            if not strict:
                self.model.load_state_dict(model_state, strict=False)
            else:
                raise

        super().load_state_dict(pipeline_state, strict=False)

    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        # ToDo: Sample volume here and cluster instance vectors

        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict, step)
        self.train()
        return model_outputs, loss_dict, metrics_dict
