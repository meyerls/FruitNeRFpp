"""
FruitNeRF implementation .
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Union
import sklearn

import torch
from torch.nn import Parameter

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
# from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.semantic_nerf_field import SemanticNerfField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    scale_gradients_by_distance_squared
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    SemanticRenderer,
    UncertaintyRenderer,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler, LinearDisparitySampler
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model
from nerfstudio.utils import colormaps
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.models.nerfacto import NerfactoModelConfig

from cf_nerf.cf_nerf_field import FruitField, FieldHeadNames

# Renderer
from torch import Tensor, nn
from jaxtyping import Float, Int

from cf_nerf.components.ray_samplers import UniformSamplerWithNoise
from torchmetrics.classification import BinaryJaccardIndex
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.viewer.viewer_elements import *
import colorsys
from random import shuffle
import copy


def HSVToRGB(h, s, v):
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b)


def getDistinctColors(n):
    huePartition = 1.0 / (n + 1)
    return (HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n))


class InstanceRenderer(nn.Module):
    """Calculate feature vector along the ray."""

    @classmethod
    def forward(
            cls,
            feature_vector: Float[Tensor, "*bs num_samples feature_dim"],
            weights: Float[Tensor, "*bs num_samples 1"],
            ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
            num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs num_classes"]:
        """Calculate feature vectors along the ray."""
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            return NotImplementedError
        else:
            return torch.sum(weights * feature_vector, dim=-2)


class RandomInstanceRenderer(nn.Module):
    """Calculate feature vector along the ray."""

    def __init__(self, feature_dim=24):
        self.color_dim = 3
        self.feature_dim = feature_dim
        self.proj_vec = torch.randn((self.feature_dim, self.color_dim))
        self.refresh_projection_matrix_button = ViewerButton(name="New Random Projection",
                                                             cb_hook=self.cb_refresh_random_projection)
        super().__init__()

    def cb_refresh_random_projection(self, handle) -> None:
        self.refresh_projection_matrix()

    def refresh_projection_matrix(self):
        self.proj_vec = torch.randn((self.feature_dim, self.color_dim))

    @classmethod
    def forward(
            cls,
            feature_vector: Float[Tensor, "*bs num_samples feature_dim"],
            weights: Float[Tensor, "*bs num_samples 1"],
            ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
            num_rays: Optional[int] = None,
            proj_vec: Optional[Tensor] = None,
    ) -> Float[Tensor, "*bs num_classes"]:
        """Calculate feature vectors along the ray."""
        if not hasattr(cls, "counter"):
            cls.counter = 0
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            return NotImplementedError
        else:
            accumulated = torch.sum(weights * feature_vector, dim=-2)
            accumulated_norm = torch.nn.functional.normalize(accumulated, dim=1).detach()

            projected = torch.einsum('pd,dc->pc', accumulated_norm,
                                     proj_vec.to(accumulated))  # instance-vec @ projec-vec

            if cls.counter % 2000 == 0:
                cls.min = projected.min(dim=0, keepdim=True).values + 1e-5
                cls.max = projected.max(dim=0, keepdim=True).values

            cls.counter += 1

            projected = projected - cls.min
            projected = projected / cls.max

            return projected


@dataclass
class CFNerfModelConfig(NerfactoModelConfig):
    """FruitModel Model Config"""

    _target: Type = field(default_factory=lambda: CFNeRFModel)
    training_steps: Dict[str, Any] = to_immutable_dict({
        'semantic': 5000,
        'instance': 10000,
    })
    """Step number when to start semantic and instance training"""
    semantic_loss_weight: float = 1.0
    pass_semantic_gradients: bool = False
    """Flag to let the gradient flow back from the semantic field to the density field"""
    pass_instance_gradients: bool = False
    """Flag to let the gradient flow back from the instance field to the density field"""
    num_layers: int = 2
    """Num Layers of the density field. """
    hidden_dim: int = 64
    """Dimension of the density field. """
    num_layers_semantic: int = 2
    """Num Layers of the semantic field. """
    hidden_dim_semantics: int = 64
    """Dimension of the semantic field. """
    geo_feat_dim: int = 15
    """Output dimension of the density field. Also size of the input feature vector"""
    num_layers_instance: int = 2
    """Num Layers of the semantic field. """
    hidden_dim_instance: int = 64
    """Dimension of the semantic field. """
    hidden_dim_transient_instance: int = 64
    """Dimension of the output vector (Before Projection Head)."""
    output_dim_instance: int = 12
    """Dimension of the output feature vector (After Projection Head)."""
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""
    temperature: float = 0.1
    """Temperature value for contrastive loss"""


class CFNeRFModel(Model):
    """FruitModel based on Nerfacto model

    Args:
        config: FruitModel configuration to instantiate model
    """

    config: CFNerfModelConfig

    def __init__(self, config: CFNerfModelConfig, metadata: Dict, **kwargs) -> None:
        assert "semantics" in metadata.keys() and isinstance(metadata["semantics"], Semantics)
        self.semantics = metadata["semantics"]
        self.test_mode = kwargs['test_mode']
        self.linear_probe = kwargs['linear_probe'][0]
        self.linear_probe_num_fruits = kwargs['linear_probe'][1]
        self.cluster_object = kwargs['cluster_object']
        self.pixel_per_pair = kwargs['pixel_per_pair']
        super().__init__(config=config, **kwargs)
        self.colormap = self.semantics.colors.clone().detach().to(self.device)
        self.counter_instance = 0
        self.color_list = list(getDistinctColors(300))
        shuffle(self.color_list)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.fruit_field = FruitField(
            self.scene_box.aabb,
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            num_layers_semantic=self.config.num_layers_semantic,
            hidden_dim_semantics=self.config.hidden_dim_semantics,
            num_layers_instance=self.config.num_layers_instance,
            hidden_dim_instance=self.config.hidden_dim_instance,
            output_dim_instance=self.config.output_dim_instance,
            hidden_dim_transient_instance=self.config.hidden_dim_transient_instance,
            log2_hashmap_size=self.config.log2_hashmap_size,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            geo_feat_dim=self.config.geo_feat_dim,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            test_mode=self.test_mode,
            num_semantic_classes=1,
            pass_semantic_gradients=self.config.pass_semantic_gradients,
        )

        if self.linear_probe:
            self.linear_probe_model = torch.nn.Linear(in_features=self.config.output_dim_instance,
                                                      out_features=self.linear_probe_num_fruits)  # + 1 if Background (we removed background from cross entropy loss)

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cuda"
        )

        # Build the proposal network(s)
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction)
            self.proposal_networks.append(network)
            self.density_fns = [network.density_fn for _ in range(self.config.num_proposal_iterations)]
        else:
            for _ in range(self.config.num_proposal_iterations):
                network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction)
                self.proposal_networks.append(network)
            self.density_fns = [network.density_fn for network in self.proposal_networks]


        # Samplers
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            update_sched=update_schedule,
            single_jitter=self.config.use_single_jitter,
        )


        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)


        #self.proposal_sampler = LinearDisparitySampler(single_jitter=self.config.use_single_jitter)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_uncertainty = UncertaintyRenderer()
        self.renderer_semantics = SemanticRenderer()
        self.renderer_instance = InstanceRenderer()
        self.random_renderer_instance = RandomInstanceRenderer(feature_dim=self.config.output_dim_instance)

        # losses
        self.rgb_loss = MSELoss()
        self.binary_cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        # metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["base_field"] = list(self.fruit_field.params['base'].parameters())
        param_groups["semantic_field"] = list(self.fruit_field.params['semantics'].parameters())
        param_groups["instance_field"] = list(self.fruit_field.params['instance'].parameters())
        if self.linear_probe:
            param_groups["linear_probe"] = list(self.linear_probe_model.parameters())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups

    def setup_inference(self, num_inference_samples):
        self.num_inference_samples = num_inference_samples  # int(200)
        self.proposal_sampler = UniformSamplerWithNoise(num_samples=self.num_inference_samples, single_jitter=False)

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks


    def get_inference_outputs(self, ray_bundle: RayBundle):
        outputs = {}

        ray_samples = self.proposal_sampler(ray_bundle)
        field_outputs = self.fruit_field.forward(ray_samples)

        outputs["rgb"] = field_outputs[FieldHeadNames.RGB]

        outputs['point_location'] = ray_samples.frustums.get_positions()
        outputs["semantics"] = field_outputs[FieldHeadNames.SEMANTICS][..., 0]
        outputs["density"] = field_outputs[FieldHeadNames.DENSITY][..., 0]
        outputs["instance"] = field_outputs[FieldHeadNames.INSTANCE]

        semantic_labels = torch.sigmoid(outputs["semantics"])
        threshold = 0.9
        semantic_labels = torch.heaviside(semantic_labels - threshold, torch.tensor(0.)).to(torch.long)

        outputs["semantics_colormap"] = semantic_labels

        return outputs

    def get_inference_outputs_old(self, ray_bundle: RayBundle):
        outputs = {}


        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        field_outputs = self.fruit_field.forward(ray_samples)

        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {"rgb": rgb,
                   "accumulation": accumulation,
                   "depth": depth,
                   "weights_list": weights_list,
                   "ray_samples_list": ray_samples_list}

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        # semantics
        semantic_weights = weights
        if not self.config.pass_semantic_gradients:
            semantic_weights = semantic_weights.detach()
        outputs["semantics"] = self.renderer_semantics(
            field_outputs[FieldHeadNames.SEMANTICS], weights=semantic_weights
        )

        # semantics colormaps
        semantic_labels = torch.sigmoid(outputs["semantics"].detach())
        threshold = 0.9
        semantic_labels = torch.heaviside(semantic_labels - threshold, torch.tensor(0.)).to(torch.long)
        outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels].repeat(1, 3)

        # instance
        instance_weights = weights
        if not self.config.pass_instance_gradients:
            instance_weights = instance_weights.detach()
        outputs["instance"] = self.renderer_instance(
            feature_vector=field_outputs[FieldHeadNames.INSTANCE],
            weights=instance_weights
        )

        outputs["instance_random"] = self.random_renderer_instance(
            feature_vector=field_outputs[FieldHeadNames.INSTANCE],
            weights=instance_weights,
            proj_vec=self.random_renderer_instance.proj_vec
        )


        if not self.training and self.cluster_object:
            # import hdbscan
            # from sklearn.metrics.pairwise import pairwise_distances

            # clusterer = self.cluster_object
            median_center = self.cluster_object['median_center']
            median_center_xyz = median_center[:, :3]
            median_center_feature = median_center[:, 3:]
            median_center_feature = np.asarray(
                torch.nn.functional.normalize(torch.asarray(median_center_feature), dim=1).cpu())

            feature_vector_normalized = np.asarray(torch.nn.functional.normalize(outputs["instance"], dim=1).cpu())
            feature_xyz = np.asarray(ray_samples.frustums.get_positions().cpu())

            cosine = 1 - np.dot(feature_vector_normalized, median_center_feature.T)

            # distance = cosine + 0.5*dist
            distance = cosine

            assigned_cluster_id = distance.argmin(axis=1)
            assigned_cluster_check = distance.min(axis=1) < 0.15

            valid_cluster_id = assigned_cluster_id[assigned_cluster_check]
            valid_cluster_idx = np.where(distance.min(axis=1) < 0.15)[0]
            invalid_semantic_idx = np.where(semantic_labels.cpu() == 0)[0]

            cluster_colors = np.zeros((feature_vector_normalized.shape[0], 3))
            cluster_colors[valid_cluster_idx] = np.asarray(self.color_list)[valid_cluster_id.astype(int)]
            cluster_colors[invalid_semantic_idx] = 0

            outputs["instance_cluster"] = torch.tensor(cluster_colors)


        return outputs

    def get_outputs(self, ray_bundle: RayBundle):
        #if self.training:
        #    self.camera_optimizer.apply_to_raybundle(ray_bundle)

        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        field_outputs = self.fruit_field.forward(ray_samples)

        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {"rgb": rgb,
                   "accumulation": accumulation,
                   "depth": depth,
                   "weights_list": weights_list,
                   "ray_samples_list": ray_samples_list}

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        # semantics
        semantic_weights = weights
        if not self.config.pass_semantic_gradients:
            semantic_weights = semantic_weights.detach()
        outputs["semantics"] = self.renderer_semantics(
            field_outputs[FieldHeadNames.SEMANTICS], weights=semantic_weights
        )

        # semantics colormaps
        semantic_labels = torch.sigmoid(outputs["semantics"].detach())
        threshold = 0.9
        semantic_labels = torch.heaviside(semantic_labels - threshold, torch.tensor(0.)).to(torch.long)
        outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels].repeat(1, 3)

        # instance
        instance_weights = weights
        if not self.config.pass_instance_gradients:
            instance_weights = instance_weights.detach()
        outputs["instance"] = self.renderer_instance(
            feature_vector=field_outputs[FieldHeadNames.INSTANCE],
            weights=instance_weights
        )

        outputs["instance_random"] = self.random_renderer_instance(
            feature_vector=field_outputs[FieldHeadNames.INSTANCE],
            weights=instance_weights,
            proj_vec=self.random_renderer_instance.proj_vec
        )

        if self.linear_probe:
            linear_probe_input = outputs["instance"].detach()
            linear_probe_input_normalized = torch.nn.functional.normalize(linear_probe_input, dim=1)
            linear_probe_output = self.linear_probe_model(linear_probe_input_normalized)
            outputs["linear_probe_multi_class_prediction"] = linear_probe_output

        if not self.training and self.cluster_object:
            # import hdbscan
            # from sklearn.metrics.pairwise import pairwise_distances

            # clusterer = self.cluster_object
            median_center = self.cluster_object['median_center']
            median_center_xyz = median_center[:, :3]
            median_center_feature = median_center[:, 3:]
            median_center_feature = np.asarray(
                torch.nn.functional.normalize(torch.asarray(median_center_feature), dim=1).cpu())

            feature_vector_normalized = np.asarray(torch.nn.functional.normalize(outputs["instance"], dim=1).cpu())
            feature_xyz = np.asarray(ray_samples.frustums.get_positions().cpu())

            cosine = 1 - np.dot(feature_vector_normalized, median_center_feature.T)

            # distance = cosine + 0.5*dist
            distance = cosine

            assigned_cluster_id = distance.argmin(axis=1)
            assigned_cluster_check = distance.min(axis=1) < 0.15

            valid_cluster_id = assigned_cluster_id[assigned_cluster_check]
            valid_cluster_idx = np.where(distance.min(axis=1) < 0.15)[0]
            invalid_semantic_idx = np.where(semantic_labels.cpu() == 0)[0]

            cluster_colors = np.zeros((feature_vector_normalized.shape[0], 3))
            cluster_colors[valid_cluster_idx] = np.asarray(self.color_list)[valid_cluster_id.astype(int)]
            cluster_colors[invalid_semantic_idx] = 0

            outputs["instance_cluster"] = torch.tensor(cluster_colors)

            # hdbscan.approximate_predict(clusterer, outputs["instance"])

        return outputs

    def contrastive_loss(self, feature_vectors, pair_position, pixel_indices_split, temperature=0.1):
        """
        Contrastive loss for instance segmentation.

        Implementation of Normalized Temperature-scaled Cross Entropy Loss (https://paperswithcode.com/method/nt-xent)

        :param feature_vectors:
        :param pair_position:
        :return:
        """

        # Reshape feature vector to num apples, num pixels, num features
        feature_vector_split = feature_vectors.view((pixel_indices_split.shape[0], pixel_indices_split.shape[1], -1))

        # temperature = 0.1
        features = feature_vector_split

        # N: Number of selected fruits(sub_cluster_neighbours * num_clusters)
        # K: Number of pixels per fruit
        # L: Feature dim
        N, K, L = features.shape

        # Normalize the feature vectors
        features = nn.functional.normalize(features, dim=-1)
        mean_features = nn.functional.normalize(features.mean(dim=1), dim=-1)  # N x L
        features = features.view(N * K, L)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, mean_features.T)  # N*K x N
        positive_mask = torch.eye(N, device=features.device).repeat_interleave(K, dim=0)  # N*K x N
        loss = torch.nn.functional.cross_entropy(similarity_matrix / temperature, positive_mask.float())
        return loss

    def contrastive_loss_old(self, feature_vectors, pair_position, pixel_indices_split, temperature=0.1):
        """
        Contrastive loss for instance segmentation.

        Implementation of Normalized Temperature-scaled Cross Entropy Loss (https://paperswithcode.com/method/nt-xent)

        :param feature_vectors:
        :param pair_position:
        :return:
        """

        # Reshape feature vector to num apples, num pixels, num features
        feature_vector_split = feature_vectors.view((pixel_indices_split.shape[0], pixel_indices_split.shape[1], -1))

        # temperature = 0.1
        features = feature_vector_split

        # N: Number of selected fruits(sub_cluster_neighbours * num_clusters)
        # K: Number of pixels per fruit
        # L: Feature dim
        N, K, L = features.shape

        # Normalize the feature vectors
        features = nn.functional.normalize(features, dim=-1)
        # Take mean of all features of all pixels of one apple and repeat it K times to have the same shape (KxN) as features
        positive_mean_features = nn.functional.normalize(features.mean(dim=1), dim=-1).unsqueeze(1).repeat(1, K, 1).view(N * K, L)
        # Reshape features to (N*K, L) for similarity computation
        features = features.view(N * K, L)

        # Compute similarity matrix -->  If feature vectors are normalized the cos-sim reduces to a dot product
        similarity_matrix = torch.matmul(features, features.T)
        # Cosine similarity for positive (fruit pixels)
        pos_similarity_matrix = torch.matmul(features, positive_mean_features.T)

        # Mask out self-similarities
        # mask = torch.eye(N * K, device=features.device, dtype=torch.bool)
        # similarity_matrix = similarity_matrix.masked_fill(mask, 0)

        # Compute the positive similarities (same instance, different pixel)
        positive_mask = torch.eye(N, device=features.device).repeat_interleave(K, dim=0).repeat_interleave(K, dim=1)
        similarity_matrix[positive_mask.to(torch.bool)] = pos_similarity_matrix[positive_mask.to(torch.bool)]
        positive_similarities = similarity_matrix[positive_mask.to(torch.bool)].view(N * K, -1)

        # log_sum_exp_nom = torch.logsumexp(positive_similarities / temperature, dim=1) / K
        log_exp_nom = torch.mean(positive_similarities / temperature, dim=1)

        # Compute log-sum-exp for normalization -> ToDo: Why - torch.log(torch.tensor(K))?
        log_sum_exp_denom = torch.logsumexp(similarity_matrix / temperature, dim=1) - torch.log(torch.tensor(K))

        # Calculate the NT-Xent loss
        loss = -log_exp_nom + log_sum_exp_denom
        loss = loss.mean()

        return loss

    def contrastive_loss_into_nce(self, feature_vectors, pair_position, pixel_indices_split, temperature=0.1):
        """
        Contrastive loss for instance segmentation.

        For ablation study

        :param feature_vectors:
        :param pair_position:
        :return:
        """

        # Reshape feature vector to num apples, num pixels, num features
        feature_vector_split = feature_vectors.view((pixel_indices_split.shape[0], pixel_indices_split.shape[1], -1))

        # temperature = 0.1
        features = feature_vector_split

        # N: Number of selected fruits(sub_cluster_neighbours * num_clusters)
        # K: Number of pixels per fruit
        # L: Feature dim
        N, K, L = features.shape

        # Normalize the feature vectors
        features = nn.functional.normalize(features, dim=-1)
        # Reshape features to (N*K, L) for similarity computation
        features = features.view(N * K, L)

        # Compute similarity matrix -->  If feature vectors are normalized the cos-sim reduces to a dot product
        similarity_matrix = torch.matmul(features, features.T)  # .to(torch.float32)

        # Mask out self-similarities
        mask = torch.eye(N * K, device=features.device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, 0)

        positive_similarities = torch.clone(similarity_matrix)
        block_mask_matrix = torch.kron(torch.eye(N), torch.ones(K, K)).to(torch.bool).to(features.device)
        # positive_similarities[~block_mask_matrix] = 0
        positive_similarities = positive_similarities[block_mask_matrix].reshape(N * K, K)

        log_exp_nom = positive_similarities.mean(dim=1) / temperature

        # Compute log-sum-exp for normalization -> ToDo: Why - torch.log(torch.tensor(K))?
        log_sum_exp_denom = torch.logsumexp(similarity_matrix / temperature, dim=1) - torch.log(torch.tensor(K))

        # Calculate the NT-Xent loss
        loss = -log_exp_nom + log_sum_exp_denom
        loss = loss.mean()

        return loss

    def get_loss_dict(self, outputs, batch, metrics_dict=None, step=-1):
        loss_dict = {}

        image = batch["image"].to(self.device)[:, :3]
        if outputs["semantics"].shape != batch['semantic'].shape:
            batch['semantic'] = batch['semantic'].unsqueeze(1).to(torch.float32)

        if self.config.training_steps['cascaded_freezing'] and step >= self.config.training_steps['instance']:
            pass
            loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"]).detach()
            loss_dict["semantics_loss"] = self.config.semantic_loss_weight * self.binary_cross_entropy_loss(
                outputs["semantics"], batch['semantic']
            )
        else:
            # image = batch["image"].to(self.device)[:, :3]
            loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])

            if step > self.config.training_steps['semantic']:
                # if outputs["semantics"].shape != batch['semantic'].shape:
                #    batch['semantic'] = batch['semantic'].unsqueeze(1).to(torch.float32)

                # Semantic Loss (BCE -> fruit or no fruit)
                loss_dict["semantics_loss"] = self.config.semantic_loss_weight * self.binary_cross_entropy_loss(
                    outputs["semantics"], batch['semantic']
                )

            if self.training:
                loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                    outputs["weights_list"], outputs["ray_samples_list"]
                )

                self.camera_optimizer.get_loss_dict(loss_dict)

        if step >= self.config.training_steps['instance'] and self.training:
            # Contrastive Loss
            loss_dict["instance_loss"] = self.contrastive_loss(
                feature_vectors=outputs["instance"],
                pair_position=batch['instances'],
                pixel_indices_split=batch['pixel_indices_split'],
                temperature=self.config.temperature
            )
            # if self.linear_probe and step + 2000 >= self.config.training_steps['instance']:
            if self.linear_probe:
                target = batch['semantics_gt'][None].clone()
                # Mask background. Feature vector are to diverse and during instance training we do not sample that much bg
                _, mask_bg = torch.where(target != 0)
                target_masked = target[0, mask_bg]

                target_one_hot = torch.nn.functional.one_hot(target_masked.long() - 1,
                                                             num_classes=self.linear_probe_model.out_features)

                input_ = outputs["linear_probe_multi_class_prediction"]
                input_masked = input_[mask_bg]

                loss_dict["linear_probe_loss"] = self.cross_entropy_loss(input_masked, target_one_hot.float())

        return loss_dict

    def forward(self, ray_bundle: RayBundle) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        if self.test_mode == 'inference':
            # fruit_nerf_output = self.get_inference_outputs(ray_bundle, self.render_rgb)
            fruit_nerf_output = self.get_inference_outputs(ray_bundle)
        else:
            fruit_nerf_output = self.get_outputs(ray_bundle)

        return fruit_nerf_output

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)[:, :3]
        # semantic_image = batch['semantic'].to(self.device)

        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        # metrics_dict["semantic_psnr"] = self.psnr(outputs['semantics'], semantic_image)

        # if self.config.
        #    metrics_dict["semantic_psnr"] = self.psnr(outputs['semantics'], semantic_image)
        metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        metrics_dict["instance_features_mean"] = torch.mean(outputs["instance"].detach())
        metrics_dict["instance_features_mean_abs"] = torch.mean(torch.abs(outputs["instance"].detach()))
        metrics_dict["instance_features_std"] = torch.std(outputs["instance"].detach())

        self.camera_optimizer.get_metrics_dict(metrics_dict)

        return metrics_dict

    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
                                     ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)[..., :3]
        rgb = outputs["rgb"]
        rgb = torch.clamp(rgb, min=0, max=1)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image[..., :3], -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb,
                       # "accumulation": combined_acc,
                       "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        # semantics
        semantic_labels = torch.sigmoid(outputs["semantics"])
        semantics_colormap = semantic_labels[..., 0].unsqueeze(-1)
        # valid mask
        semantic_gt = batch["semantic"].to(self.device).unsqueeze(-1)

        metric = BinaryJaccardIndex().to(self.device)
        iou = metric(semantic_labels[..., 0], batch["semantic"])
        metrics_dict["iou-semantic"] = float(iou)

        combined_semantic = torch.cat([semantic_gt, semantics_colormap], dim=1)
        images_dict["semantic"] = combined_semantic

        images_dict["instance_random"] = outputs["instance_random"]

        instance_random_image = outputs["instance_random"].clone()
        instance_mask = semantics_colormap[..., 0] > 0.85
        instance_random_image[..., :][~instance_mask] = 0

        images_dict["instance_random_semantic_masked"] = instance_random_image

        if not self.training and self.linear_probe:
            target = batch['semantics_gt'].reshape(-1)[None].clone()

            # Mask background. Feature vector are to diverse and during instance training we do not sample that much bg
            _, mask_bg = torch.where(target.detach() != 0)

            input_ = outputs["linear_probe_multi_class_prediction"].reshape(-1, self.linear_probe_model.out_features)
            input_ = input_[mask_bg]

            predicted_label = torch.argmax(input_, dim=1)
            target_ids_fruit = target[0][mask_bg].cpu() - 1

            f1 = sklearn.metrics.f1_score(target_ids_fruit, y_pred=predicted_label.cpu(), average='micro')
            recall = sklearn.metrics.recall_score(target_ids_fruit, y_pred=predicted_label.cpu(), average='micro')
            precision = sklearn.metrics.precision_score(target_ids_fruit, y_pred=predicted_label.cpu(), average='micro')

            metrics_dict["Linear_probe/F1_pixel_wise_wo_bg"] = f1
            metrics_dict["Linear_probe/Recall_pixel_wise_wo_bg"] = recall
            metrics_dict["Linear_probe/Precision_pixel_wise_wo_bg"] = precision

            predicted_fruit_ids = torch.unique(predicted_label).cpu()
            target_fruit_ids = torch.unique(target_ids_fruit) - 1

            matches = target_fruit_ids.repeat(predicted_fruit_ids.shape[0], 1) == predicted_fruit_ids[None].T
            TP = matches.sum()
            FP = torch.abs(predicted_fruit_ids.shape[0] - TP) + 1e-12
            FN = target_fruit_ids.shape[0] - TP

            Precision = TP / (TP + FP)
            Recall = TP / (TP + FN)
            F1 = 2 * Precision * Recall / (Precision + Recall)

            metrics_dict["Linear_probe/F1_fruit_count_per_image"] = F1
            metrics_dict["Linear_probe/Recall_fruit_count_per_image"] = Recall
            metrics_dict["Linear_probe/Precision_fruit_count_per_image"] = Precision

        return metrics_dict, images_dict
