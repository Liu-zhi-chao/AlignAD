from dataclasses import dataclass
from typing import Tuple

import numpy as np
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.maps.abstract_map import SemanticMapLayer


@dataclass
class AlignADConfig:
    cache_data: bool = False

    b2d: bool = False   # Bench2Drive
    agent_pred: bool=False
    area_pred: bool=False

    navsim_traj_20_path = "pretrained/kmeans_navsim_traj_20.npy"

    refiner_num: int = 3
    exp_only: bool = False

    proposal_num: int = 64
    score_num: int = 6
    point_cloud_range = [-32, -32, -2.0, 32, 32, 6.0]
    num_points_in_pillar: int = 4
    num_learnable_pts: int = 0

    half_length: float = 2.588 + 0.25 #small buffer for safety
    half_width: float = 1.1485 + 0.1
    rear_axle_to_center: float = 1.461
    ego_scale = [half_length*2, half_width*2]
    lidar_height: float = 0

    num_poses: int = 8
    command_num: int= 4
    ####### image backbone #######
    img_backbone = "resnet34"
    resnet34_ckpt = "pretrained/pytorch_model.bin"
    ####### lidar backbone #######
    embed_dims = 512
    lidar_backbone = dict(
        # type="ResNet",
        depth=34,
        in_channels=1,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN2d", requires_grad=True),
        # norm_eval=False,
        style="pytorch",
    )
    ####### lidar fpn ###########
    level_num = 3
    lidar_neck = dict(
        type="FPN",
        in_channels=[128, 256, 512],
        out_channels=embed_dims,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=level_num,
        relu_before_extra_convs=True,
    )

    image_shape_raw = [1080, 1920]
    include_opa = False
    semantics = True
    semantic_dim = 7
    scale_range = [0.01, 3.2]
    unit_xy = [4.0, 4.0]
    xy_coordinate = "cartesian"
    phi_activation = "sigmoid"
    pc_range = point_cloud_range

    # Transformer
    tf_d_model: int = embed_dims
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0
    num_bev_layers: int=1
    
    # loss weights
    prev_weight: float = 0.1
    trajectory_weight: float = 1.
    final_score_weight: float = 1.
    sub_score_weight: float = 0.
    bev_semantic_weight: float = 1.
    pred_ce_weight: int = 1
    pred_l1_weight: int = 0.1
    pred_area_weight: int = 2
    
    # Denoising alignment configuration
    enable_denoising: bool = True  # Enable noise-injected denoising alignment
    denoise_loss_weight: float = 1.0  # Overall weight for denoising loss in total loss
    denoise_noise_scale: Tuple[float, float] = (2.0, 2.0)  # Max noise (dx, dy) in meters for injection
    denoise_yaw_scale: float = 0.1  # Max yaw noise in radians
    denoise_include_yaw: bool = False  # Whether to include yaw in noise
    denoise_sys_scale: float = 0.2  # Max magnitude for systematic noise (||delta_sys|| ∈ [0, sys_scale])
    denoise_proj_channels: int = 128  # Projection dimension for InfoNCE
    denoise_hidden_channels: int = 256  # Hidden dimension for denoiser MLP
    denoise_infonce_tau: float = 0.1  # Temperature for InfoNCE
    denoise_infonce_radius: float = 1.0  # Radius for positive samples in meters
    denoise_infonce_topk: int = 8  # Top-K nearest neighbors as positives (for 64*8 points, 8-16 is reasonable)
    denoise_lambda_inj: float = 1.0  # Weight for injection noise regression loss (delta_inj vs eps_gt)
    denoise_lambda_noise: float = 1.0  # Backward compat alias for lambda_inj
    denoise_lambda_nce: float = 0.1  # Weight for InfoNCE loss (post-correction)
    denoise_lambda_imp: float = 1.0  # Weight for improvement margin loss
    denoise_margin: float = 0.1  # Margin for improvement loss (L_nce_post should be < L_nce_pre - margin)
    denoise_lambda_reg: float = 0.0  # Deprecated, kept for backward compatibility

    #others
    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    # new
    lidar_seq_len: int = 1

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp = 4
    n_layer = 2  # Number of transformer layers used in the vision backbone
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True

    # detection
    num_bounding_boxes: int = 30

    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height
    bev_pixel_size: float = 0.25

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2
    
    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
