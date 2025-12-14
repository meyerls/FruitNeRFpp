"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from cf_nerf.cf_nerf_datamanager import (
    CFNeRFDataManagerConfig,
)
from cf_nerf.cf_nerf_model import CFNerfModelConfig
from cf_nerf.template_nerf_model import TemplateModelConfig
from cf_nerf.cf_nerf_pipeline import (
    CFNeRFPipelineConfig,
)
from cf_nerf.components.cf_nerf_dataparser import CFNeRFDataParserConfig
from cf_nerf.components.pixel_samplers import ImageContrastivePixelSamplerConfig, SemanticPixelSamplerConfig, \
    PixelSamplerConfig, ClusterContrastivePixelSamplerConfig
from cf_nerf.cf_nerf_trainer import CFNeRFTrainerConfig, CFNeRFTrainer

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig, MultiStepSchedulerConfig
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.configs.base_config import MachineConfig
import torch
import random
import numpy as np

# random.seed(42)
# torch.manual_seed(42)
# np.random.seed(42)
# torch.use_deterministic_algorithms(True)


cf_nerf_small = MethodSpecification(
    config=CFNeRFTrainerConfig(
        method_name="cf-nerf-small",
        steps_per_eval_batch=500,
        steps_per_eval_image=500,
        steps_per_eval_all_images=40000,
        steps_per_save=2000,
        max_num_iterations=100000,
        mixed_precision=True,
        ignore_instance_weights=False,
        save_only_latest_checkpoint=False,
        pipeline=CFNeRFPipelineConfig(
            datamanager=CFNeRFDataManagerConfig(
                train_num_images_to_sample_from=200,
                train_num_times_to_repeat_images=1000,
                rgb_pixel_sampler=PixelSamplerConfig(),
                semantic_pixel_sampler=SemanticPixelSamplerConfig(bg_to_instance_ratio=0.95),
                instance_pixel_sampler=ClusterContrastivePixelSamplerConfig(num_clusters=8, num_fruits_per_cluster=4),
                train_num_rays_per_batch=2048 * 2,  # Default: 2048
                eval_num_rays_per_batch=1024,  # Default: 2048
            ),
            model=CFNerfModelConfig(
                # training_steps={'semantic': 50000, 'instance': 150000, 'cascaded_freezing': True},# Default
                training_steps={'semantic': 5000, 'instance': 7000, 'cascaded_freezing': True},  # Default
                num_nerf_samples_per_ray=64,
                log2_hashmap_size=19,
                num_layers=2,
                hidden_dim=32,
                hidden_dim_color=32,
                appearance_embed_dim=32,
                geo_feat_dim=24,
                num_layers_semantic=2,
                hidden_dim_semantics=32,
                temperature=0.2,
                num_layers_instance=5,
                hidden_dim_instance=32,
                hidden_dim_transient_instance=32,
                output_dim_instance=12,
                average_init_density=0.01,
                eval_num_rays_per_chunk=1 << 13,
            ),
        ),
        optimizers={
            "proposal_networks": {"optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                                  "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
                                  },
            "base_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=80000),
            },
            "semantic_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=120000, warmup_steps=50000),
            },
            "instance_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=300000, warmup_steps=150000),
            },
            "linear_probe": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-13, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-15, max_steps=50000, warmup_steps=10000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 13),
        vis="viewer",
        machine=MachineConfig(seed=42)
    ),
    description="CF-NeRF: FruitNeRF with Contrastive Learning.",
)

cf_nerf = MethodSpecification(  # cf_nerf
    config=CFNeRFTrainerConfig(
        method_name="cf-nerf",
        steps_per_eval_batch=500,
        steps_per_eval_image=500,
        steps_per_eval_all_images=20_000,
        steps_per_save=20_000,
        max_num_iterations=250_000,
        mixed_precision=True,
        ignore_instance_weights=False,
        save_only_latest_checkpoint=False,
        pipeline=CFNeRFPipelineConfig(
            datamanager=CFNeRFDataManagerConfig(
                train_num_images_to_sample_from=200,
                train_num_times_to_repeat_images=1000,
                rgb_pixel_sampler=PixelSamplerConfig(),
                semantic_pixel_sampler=PixelSamplerConfig(),
                instance_pixel_sampler=ClusterContrastivePixelSamplerConfig(num_clusters=8),
                train_num_rays_per_batch=2048,  # Default: 4096
                eval_num_rays_per_batch=2048,  # Default: 4096
                ),
            model=CFNerfModelConfig(
                training_steps={'semantic': 80_000, 'instance': 120_000, 'cascaded_freezing': True},
                num_nerf_samples_per_ray=128,
                num_proposal_samples_per_ray=(512, 256),
                log2_hashmap_size=21,
                num_layers=2,
                hidden_dim=128,
                hidden_dim_color=128,
                appearance_embed_dim=128,
                geo_feat_dim=30,
                num_layers_semantic=3,
                hidden_dim_semantics=128,
                temperature=0.3,
                num_layers_instance=5,
                hidden_dim_instance=128,
                hidden_dim_transient_instance=128,
                output_dim_instance=32,
                average_init_density=0.01,
                eval_num_rays_per_chunk=1 << 14,
                proposal_weights_anneal_max_num_iters=5000,
                max_res=4096,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=250_000),
            },
            "base_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=100_000),
            },
            "semantic_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=100_000, warmup_steps=80_000),
            },
            "instance_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=180_000, warmup_steps=120_000),
            },
            "linear_probe": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": RAdamOptimizerConfig(lr=4e-16, eps=1e-8, weight_decay=1e-3),
                "scheduler": None,
            }

        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
        # machine=MachineConfig(seed=42)
    ),
    description="CF-NeRF: FruitNeRF with Contrastive Learning.",
)

cf_nerf_fuji = MethodSpecification(  # cf_nerf_fuji
    config=CFNeRFTrainerConfig(
        method_name="cf-nerf-fuji",
        steps_per_eval_batch=500,
        steps_per_eval_image=500,
        steps_per_eval_all_images=20_000,
        steps_per_save=20_000,
        max_num_iterations=350_000,
        mixed_precision=True,
        ignore_instance_weights=False,
        save_only_latest_checkpoint=False,
        pipeline=CFNeRFPipelineConfig(
            datamanager=CFNeRFDataManagerConfig(
                train_num_images_to_sample_from=200,
                train_num_times_to_repeat_images=1000,
                rgb_pixel_sampler=PixelSamplerConfig(),
                semantic_pixel_sampler=PixelSamplerConfig(),
                instance_pixel_sampler=ClusterContrastivePixelSamplerConfig(num_clusters=4, num_fruits_per_cluster=4),
                train_num_rays_per_batch=2048 * 2,  # Default: 4096
                eval_num_rays_per_batch=2048,  # Default: 4096
            ),
            model=CFNerfModelConfig(
                training_steps={'semantic': 80_000, 'instance': 120_000, 'cascaded_freezing': True},
                num_nerf_samples_per_ray=128,
                num_proposal_samples_per_ray=(512, 256),
                log2_hashmap_size=21,
                num_layers=2,
                hidden_dim=128,
                hidden_dim_color=128,
                appearance_embed_dim=128,
                geo_feat_dim=30,
                num_layers_semantic=3, # 2
                hidden_dim_semantics=128,
                temperature=0.2,  # 0.1
                num_layers_instance=5,
                hidden_dim_instance=128,
                hidden_dim_transient_instance=128,
                output_dim_instance=32,
                average_init_density=0.01,
                eval_num_rays_per_chunk=1 << 13,
                proposal_weights_anneal_max_num_iters=5000,
                max_res=8192,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=250_000),
            },
            "base_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=120_000),
            },
            "semantic_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=100_000, warmup_steps=80_000),
            },
            "instance_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=180_000, warmup_steps=120_000),
            },
            "linear_probe": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-8, weight_decay=1e-3),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-16, max_steps=80_000, warmup_steps=5000),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
        machine=MachineConfig(seed=42)
    ),
    description="CF-NeRF: FruitNeRF with Contrastive Learning.",
)

cf_nerf_synthetic_cluster = MethodSpecification(
    config=CFNeRFTrainerConfig(
        method_name="cf-nerf-synthetic-cluster",
        steps_per_eval_batch=2000,
        steps_per_eval_image=2000,
        steps_per_eval_all_images=20_000,
        steps_per_save=5000,
        max_num_iterations=1,
        mixed_precision=True,
        ignore_instance_weights=False,
        save_only_latest_checkpoint=True,
        pipeline=CFNeRFPipelineConfig(
            datamanager=CFNeRFDataManagerConfig(
                train_num_images_to_sample_from=200,
                train_num_times_to_repeat_images=1000,
                rgb_pixel_sampler=PixelSamplerConfig(),
                semantic_pixel_sampler=PixelSamplerConfig(),
                instance_pixel_sampler=ClusterContrastivePixelSamplerConfig(num_clusters=8, num_fruits_per_cluster=4),
                train_num_rays_per_batch=1,  # Default: 4096
                eval_num_rays_per_batch=1,  # Default: 4096
            ),
            model=CFNerfModelConfig(
                training_steps={'semantic': 1, 'instance': 1, 'cascaded_freezing': True},
                num_nerf_samples_per_ray=128,
                num_proposal_samples_per_ray=(512, 256),
                log2_hashmap_size=21,
                num_layers=2,
                hidden_dim=128,
                hidden_dim_color=128,
                appearance_embed_dim=128,
                geo_feat_dim=30,
                num_layers_semantic=2,
                hidden_dim_semantics=64,
                temperature=0.2,
                num_layers_instance=5,
                hidden_dim_instance=128,
                hidden_dim_transient_instance=128,
                output_dim_instance=32,
                average_init_density=0.01,
                eval_num_rays_per_chunk=1 << 15,
                proposal_weights_anneal_max_num_iters=5000,
                max_res=4096,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "base_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=1),
            },
            "semantic_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=1, warmup_steps=1),
            },
            "instance_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=1, warmup_steps=1),
            },
            "linear_probe": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": RAdamOptimizerConfig(lr=4e-16, eps=1e-8, weight_decay=1e-3),
                "scheduler": None,
            }

        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
        machine=MachineConfig(seed=42)
    ),
    description="CF-NeRF: FruitNeRF with Contrastive Learning.",
)

"""
CF-NeRF Tree 02

cf_nerf = MethodSpecification(
    config=CFNeRFTrainerConfig(
        method_name="cf-nerf",
        steps_per_eval_batch=500,
        steps_per_eval_image=500,
        steps_per_eval_all_images=20_000,
        steps_per_save=20_000,
        max_num_iterations=250_000,
        mixed_precision=True,
        ignore_instance_weights=False,
        save_only_latest_checkpoint=False,
        pipeline=CFNeRFPipelineConfig(
            datamanager=CFNeRFDataManagerConfig(
                rgb_pixel_sampler=PixelSamplerConfig(),
                semantic_pixel_sampler=SemanticPixelSamplerConfig(bg_to_instance_ratio=0.95),
                instance_pixel_sampler=ClusterContrastivePixelSamplerConfig(num_clusters=8),
                dataparser=CFNeRFDataParserConfig(
                    train_split_fraction=0.99
                ),
                train_num_rays_per_batch=2048 * 2,  # Default: 4096
                eval_num_rays_per_batch=2048,  # Default: 4096
            ),
            model=CFNerfModelConfig(
                training_steps={'semantic': 80_000, 'instance': 120_000, 'cascaded_freezing': True},
                num_nerf_samples_per_ray=128,
                num_proposal_samples_per_ray=(512, 256),
                log2_hashmap_size=21,
                num_layers=2,
                hidden_dim=64,
                hidden_dim_color=64,
                appearance_embed_dim=64,
                geo_feat_dim=24,
                num_layers_semantic=2,
                hidden_dim_semantics=64,
                temperature=0.2,
                num_layers_instance=5,
                hidden_dim_instance=128,
                hidden_dim_transient_instance=128,
                output_dim_instance=16,
                average_init_density=0.01,
                eval_num_rays_per_chunk=1 << 14,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=250_000),
            },
            "base_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=100_000),
            },
            "semantic_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=100_000, warmup_steps=80_000),
            },
            "instance_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200_000, warmup_steps=120_000),
            },
            "linear_probe": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-13, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-14, max_steps=5000),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
        # machine=MachineConfig(seed=42)
    ),
    description="CF-NeRF: FruitNeRF with Contrastive Learning.",
)"""

'''
Config for Cluster:


cf_nerf_cluster = MethodSpecification(
    config=CFNeRFTrainerConfig(
        method_name="cf-nerf-cluster",
        steps_per_eval_batch=500,
        steps_per_eval_image=500,
        steps_per_eval_all_images=30000,
        steps_per_save=2000,
        max_num_iterations=150000,
        mixed_precision=True,
        pipeline=CFNeRFPipelineConfig(
            datamanager=CFNeRFDataManagerConfig(
                rgb_pixel_sampler=PixelSamplerConfig(),
                semantic_pixel_sampler=PixelSamplerConfig(),
                instance_pixel_sampler=ClusterContrastivePixelSamplerConfig(),
                dataparser=CFNeRFDataParserConfig(
                    train_split_fraction=0.99
                ),
                train_num_rays_per_batch=4096 * 2,
                eval_num_rays_per_batch=2048,
            ),
            model=CFNerfModelConfig(
                training_steps={'semantic': 20000, 'instance': 50000, 'cascaded_freezing': True},
                num_nerf_samples_per_ray=128,
                num_proposal_samples_per_ray=(512, 256),
                log2_hashmap_size=21,
                num_layers=2,
                hidden_dim=128,
                hidden_dim_color=64,
                appearance_embed_dim=64,
                geo_feat_dim=24,
                num_layers_semantic=2,
                hidden_dim_semantics=32,
                temperature=0.2,
                num_layers_instance=5,
                hidden_dim_instance=128,
                hidden_dim_transient_instance=256,
                output_dim_instance=12,
                average_init_density=0.01,
                eval_num_rays_per_chunk=1 << 13,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None  # ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "base_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-4, max_steps=80000),
            },
            "semantic_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=40000, warmup_steps=20000),
            },
            "instance_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=150000, warmup_steps=50000),
            },
            "linear_probe": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=15000, warmup_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 13),
        vis="viewer",
        machine=MachineConfig(seed=42)
    ),
    description="CF-NeRF: FruitNeRF with Contrastive Learning.",
)

cf-nerf-small
--data
/home/se86kimy/Dropbox/07_data/MessyRoom/large_corridor_100/messy_room --vis wandb --viewer.camera-frustum-scale 0.2 --pipeline.datamanager.train-num-images-to-sample-from 200 --pipeline.datamanager.eval-num-images-to-sample-from 3 --pipeline.datamanager.train-num-times-to-repeat-images 2000

instance-pointcloud --load-config /home/woody/iwi9/iwi9019h/data/messy_room_results/messy_room/cf-nerf-fuji/2024-09-09_204500/config.yml --output-dir /home/woody/iwi9/iwi9019h/data/messy_room_results/messy_room/cf-nerf-fuji/2024-09-09_204500 --use-bounding-box True --use-bounding-box True --bounding-box-min -1 -1 -1 --bounding-box-max 1  1 1 --num_rays_per_batch 4096 --num_points_per_side 1_500

--load_pcd /home/woody/iwi9/iwi9019h/data/messy_room_results/messy_room/cf-nerf-fuji/2024-09-09_204500/cf-nerf-fuji --output_dir /home/woody/iwi9/iwi9019h/data/messy_room_results/messy_room/cf-nerf-fuji/2024-09-09_204500/cf-nerf-fuji --staged-num-clusters 40 --clustering_variant staged --staged_max_points 600000

ns-train cf-nerf-fuji --data /home/woody/iwi9/iwi9019h/data/FUJI/FUJI_FruitNeRF_GT --viewer.camera-frustum-scale 0.2 --vis wandb --output-dir /home/woody/iwi9/iwi9019h/data/FUJI_results

'''
