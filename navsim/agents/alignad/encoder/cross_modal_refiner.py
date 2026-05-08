import torch.nn as nn
import numpy as np
import torch
from typing import Dict, Optional, Tuple
from .encoder import BEVFormerEncoder, BEVFormerEncoderWithDenoising
from ..alignad_config import AlignADConfig
from .blocks import linear_relu_ln, gen_sineembed_for_position
from .spatial_cross_attention import MSDeformableAttention3D
from .temporal_self_attention import TemporalSelfAttention
from .decoder import CustomMSDeformableAttention
from .denoising_alignment import DenoisingAlignmentModule
from mmdet.models.utils.positional_encoding import LearnedPositionalEncoding
from .ref2d_updater import Ref2DSoftTopKUpdater

class CrossModalRefiner(nn.Module):
    def __init__(self, config: AlignADConfig, enable_denoising: bool = False):
        super().__init__()
        self.proposal_num = config.proposal_num
        self.pose_num = config.num_poses
        self.pose_dim=3
        self.enable_denoising = enable_denoising

        d_model = config.tf_d_model
        d_ffn = config.tf_d_ffn
        self.embed_dim = d_model

        num_points = config.num_points_in_pillar * 4

        _num_levels_ = 3
        num_cams = 3

        num_layers = config.num_bev_layers

        half_length = config.half_length
        half_width = config.half_width
        rear_axle_to_center = config.rear_axle_to_center
        lidar_height=config.lidar_height

        self.pc_range = np.array(config.point_cloud_range)
        
        self.positional_encoding = nn.Sequential(
                                            *linear_relu_ln(d_model, 1, 1, d_model),
                                            nn.Linear(d_model, d_model),
                                        )
        # load anchors: shape [20, 8, 2] (x, y only, no heading)
        navsim_traj_20_path = config.navsim_traj_20_path
        anchors_np = np.load(navsim_traj_20_path)  # [64, 8, 2]
        self.register_buffer("anchors", torch.from_numpy(anchors_np).float())  # [64, 8, 2]
        
        self.ref2d_updater = Ref2DSoftTopKUpdater(
            embed_dim=d_model,
            k=2,
            tau=0.5,
            hidden=256,
            num_proposals=config.proposal_num,
            num_poses=config.num_poses,
            state_dim=2,
            anchors=self.anchors,
        )
        
        # Get denoising config from AlignADConfig if available
        noise_scale = getattr(config, 'denoise_noise_scale', (2.0, 2.0))
        include_yaw = getattr(config, 'denoise_include_yaw', False)
        sys_scale = getattr(config, 'denoise_sys_scale', 0.2)
        infonce_radius = getattr(config, 'denoise_infonce_radius', 1.0)
        infonce_topk = getattr(config, 'denoise_infonce_topk', 8)
        infonce_tau = getattr(config, 'denoise_infonce_tau', 0.1)
        proj_channels = getattr(config, 'denoise_proj_channels', 128)
        hidden_channels = getattr(config, 'denoise_hidden_channels', 256)
        lambda_inj = getattr(config, 'denoise_lambda_inj', 1.0)
        lambda_noise = getattr(config, 'denoise_lambda_noise', 1.0)  # Backward compat
        lambda_nce = getattr(config, 'denoise_lambda_nce', 0.1)
        lambda_imp = getattr(config, 'denoise_lambda_imp', 0.1)
        margin = getattr(config, 'denoise_margin', 0.1)
        
        # Choose encoder based on denoising flag
        encoder_class = BEVFormerEncoderWithDenoising if enable_denoising else BEVFormerEncoder
        
        encoder_kwargs = dict(
            embed_dims=d_model,
            num_layers=num_layers,
            pc_range=self.pc_range,
            num_points_in_pillar=config.num_points_in_pillar,
            num_learnable_pts=config.num_learnable_pts,
            half_length=half_length,
            half_width=half_width,
            rear_axle_to_center=rear_axle_to_center,
            lidar_height=lidar_height,
            return_intermediate=False,
            transformerlayers=dict(
                type='BEVFormerLayerWithDenoising' if enable_denoising else 'BEVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='SelfAttention',
                        embed_dims=d_model,
                        num_heads=config.tf_num_head,
                        attn_drop=config.tf_dropout,
                        proj_drop=config.tf_dropout,
                        is_global=True,
                       ),
                    dict(
                        type='SpatialCrossAttention',
                        num_cams=1,
                        pc_range=config.point_cloud_range,
                        dropout=config.tf_dropout,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=d_model,
                            num_points=num_points,
                            num_levels=3,
                            use_offset=True),
                        embed_dims=d_model,
                    ),
                    dict(
                        type='SpatialCrossAttention',
                        num_cams=num_cams,
                        pc_range=config.point_cloud_range,
                        dropout=config.tf_dropout,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=d_model,
                            num_points=num_points,
                            num_levels=_num_levels_,
                            use_offset=True),
                        embed_dims=d_model,
                    ),
                ],
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=d_model,
                    feedforward_channels=config.tf_d_ffn,
                    num_fcs=2,
                    ffn_drop=config.tf_dropout,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                feedforward_channels=d_ffn,
                ffn_dropout=config.tf_dropout,
                operation_order=('self_attn', 'norm', 'cross_attn_pts', 'norm', 'cross_attn_img', 'norm', 'ffn', 'norm')),
        )
        
        # Add denoising-specific kwargs (noise decomposition + improvement margin)
        if enable_denoising:
            encoder_kwargs.update(dict(
                noise_scale=noise_scale,
                include_yaw=include_yaw,
                sys_scale=sys_scale,
                proj_channels=proj_channels,
                denoise_hidden=hidden_channels,
                infonce_tau=infonce_tau,
                infonce_radius=infonce_radius,
                infonce_topk=infonce_topk,
                lambda_inj=lambda_inj,
                lambda_noise=lambda_noise,
                lambda_nce=lambda_nce,
                lambda_imp=lambda_imp,
                margin=margin,
            ))
        
        self.bev_decoder = encoder_class(**encoder_kwargs)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

    def forward(
        self, 
        image_feature, 
        pts_feature, 
        proposal, 
        bev_query, 
        *args, 
        training: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for cross-modal refinement.
        
        Args:
            image_feature: Tuple of (feat_flatten_img, spatial_shapes_img, level_start_index_img, kwargs)
            pts_feature: Tuple of (feat_flatten_pts, spatial_shapes_pts, level_start_index_pts)
            proposal: Trajectory proposals [B, num_proposals, num_poses, 3]
            bev_query: BEV query features [B, N, C]
            training: Whether in training mode (for noise injection)
            
        Returns:
            bev_query: Updated BEV query features [B, N, C]
            denoising_output: Dictionary containing denoising alignment outputs and losses
        """
        batch_size = proposal.shape[0]
        trajectory_pose = proposal.reshape(batch_size, -1, self.pose_dim)   # bs, 64*8, 3
        ref_2d = trajectory_pose.detach()

        ref_2d_xy = ref_2d[..., :2]  # [B, 512, 2]
        ref_2d_heading = ref_2d[..., 2:3]  # [B, 512, 1]
        ref_2d_xy_updated = self.ref2d_updater(bev_query, ref_2d_xy, anchors=self.anchors)  # [B, 512, 2]
        ref_2d = torch.cat([ref_2d_xy_updated, ref_2d_heading], dim=-1)  # [B, 512, 3]

        pos_embedding = gen_sineembed_for_position(trajectory_pose[...,:2], hidden_dim=self.embed_dim)
        pos_embedding = self.positional_encoding(pos_embedding)

        feat_flatten_img, spatial_shapes_img, level_start_index_img, img_kwargs = image_feature
        feat_flatten_pts, spatial_shapes_pts, level_start_index_pts = pts_feature

        # Merge kwargs
        merged_kwargs = {**kwargs, **img_kwargs}
        
        if self.enable_denoising:
            # Use denoising encoder
            bev_query, denoising_output = self.bev_decoder(
                bev_query,
                feat_flatten_img,
                feat_flatten_pts,
                bev_pos=pos_embedding,
                spatial_shapes_img=spatial_shapes_img,
                level_start_index_img=level_start_index_img,
                spatial_shapes_pts=spatial_shapes_pts,
                level_start_index_pts=level_start_index_pts,
                bev_h=self.proposal_num,
                bev_w=self.pose_num,
                ref_2d=ref_2d,
                training=training,
                **merged_kwargs
            )
        else:
            # Use standard encoder
            bev_query = self.bev_decoder(
                bev_query,
                feat_flatten_img,
                feat_flatten_pts,
                bev_pos=pos_embedding,
                spatial_shapes_img=spatial_shapes_img,
                level_start_index_img=level_start_index_img,
                spatial_shapes_pts=spatial_shapes_pts,
                level_start_index_pts=level_start_index_pts,
                bev_h=self.proposal_num,
                bev_w=self.pose_num,
                ref_2d=ref_2d,
                **merged_kwargs
            )
            denoising_output = {}

        return bev_query, denoising_output