from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from typing import Dict, Optional, Tuple
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
import torch.nn as nn
from .utils import safe_sigmoid
from .denoising_alignment import DenoisingAlignmentModule, ApplyGlobalNoise

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, embed_dims, pc_range=None, num_points_in_pillar=4, num_learnable_pts=5, lidar_height=0, 
                 half_width=0, half_length=0,rear_axle_to_center=0, return_intermediate=False,
                 **kwargs):
        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_corners = 4
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.embed_dims = embed_dims

        self.num_learnable_pts = num_learnable_pts
        if num_learnable_pts > 0:
            self.tj_learnable_fc = nn.Linear(self.embed_dims, num_learnable_pts * 2)
            self.gs_learnable_fc = nn.Linear(self.embed_dims, num_learnable_pts * 2)
        self.half_length = half_length
        self.half_width = half_width
        self.rear_axle_to_center = rear_axle_to_center
        self.lidar_height=lidar_height

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cpu', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z#4,8,8
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W#4,8,8
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H#4,8,8
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 2, 1).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)#1,4,64,3
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_x,ref_y = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, img_metas):
        # lidar2img: (B, num_cam, 4, 4)
        lidar2img = img_metas['lidar2img']
        num_cam = lidar2img.size(1)  # Number of cameras
        D, B, num_query = reference_points.size()[:3]  # D: depth, B: batch size, num_query: number of queries

        # reference_points: (D, B, 1, num_query, 4) -> (D, B, num_cam, num_query, 4, 1)
        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        # lidar2img: (1, B, num_cam, 1, 4, 4) -> (D, B, num_cam, num_query, 4, 4)
        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        # reference_points_cam: (D, B, num_cam, num_query, 4, 1) -> (D, B, num_cam, num_query, 4)
        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)

        eps = 1e-5

        # bev_mask: (D, B, num_cam, num_query, 1)
        bev_mask = (reference_points_cam[..., 2:3] > eps)

        # reference_points_cam: (D, B, num_cam, num_query, 2)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        # Normalize reference_points_cam
        reference_points_cam[..., 0] /= img_metas['img_shape'][0][0][1]  # Normalize x-coordinates
        reference_points_cam[..., 1] /= img_metas['img_shape'][0][0][0]  # Normalize y-coordinates

        # Update bev_mask: (D, B, num_cam, num_query, 1)
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        # Handle NaN values in bev_mask
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        # reference_points_cam: (D, B, num_cam, num_query, 2) -> (num_cam, B, num_query, D, 2)
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)

        # bev_mask: (D, B, num_cam, num_query, 1) -> (num_cam, B, num_query, D)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    def point_sampling_pts(self, reference_points):

        num_cam = 1
        D, B, num_query = reference_points.size()[:3]

        reference_points = reference_points.view(
            D, B, 1, num_query, 2).repeat(1, 1, num_cam, 1, 1)

        reference_points_pts = reference_points[..., :2]
        reference_points_pts[..., 0] = (reference_points_pts[..., 0] - self.pc_range[0]) / (
            self.pc_range[3] - self.pc_range[0]
        )
        reference_points_pts[..., 1] = (reference_points_pts[..., 1] - self.pc_range[1]) / (
            self.pc_range[4] - self.pc_range[1]
        )

        bev_mask = (
            (reference_points_pts[..., 0] >= 0)
            & (reference_points_pts[..., 1] >= 0)
            & (reference_points_pts[..., 0] < 1)
            & (reference_points_pts[..., 1] < 1)
        )

        reference_points_pts = reference_points_pts.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0)

        return reference_points_pts, bev_mask

    def compute_corners(self, instance_feature, boxes):
        B, P, C = instance_feature.size()
        # Calculate half dimensions
        x = boxes[..., 0]        # x-coordinate of the center
        y = boxes[..., 1]        # y-coordinate of the center
        headings= boxes[..., 2]

        half_width = torch.zeros_like(x)+self.half_width
        half_length = torch.zeros_like(x)+self.half_length

        cos_yaw = torch.cos(headings)[...,None]
        sin_yaw = torch.sin(headings)[...,None]

        x=x[...,None]+self.rear_axle_to_center * cos_yaw
        y=y[...,None]+self.rear_axle_to_center * sin_yaw

        # Compute the four corners
        corners_x = torch.stack( [half_length, half_length, -half_length, -half_length],dim=-1)
        corners_y = torch.stack( [half_width, -half_width, -half_width, half_width],dim=-1)
        # corners_x = torch.stack([torch.zeros_like(half_length), half_length, half_length, -half_length, -half_length],dim=-1)
        # corners_y = torch.stack([torch.zeros_like(half_length), half_width, -half_width, -half_width, half_width],dim=-1)

        
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                safe_sigmoid(
                    self.tj_learnable_fc(instance_feature).reshape(
                        B, P, self.num_learnable_pts, 2
                    )
                )
                - 0.5
            )
            corners_x = torch.cat([corners_x, learnable_scale[..., 0]], dim=-1)
            corners_y = torch.cat([corners_y, learnable_scale[..., 1]], dim=-1)

        # Rotate corners by yaw
        rot_corners_x = cos_yaw * corners_x + (-sin_yaw) * corners_y
        rot_corners_y = sin_yaw * corners_x + cos_yaw * corners_y

        # Translate corners to the center of the bounding box
        corners = torch.stack((rot_corners_x + x, rot_corners_y + y), dim=-1)

        return corners

    @auto_fp16()
    def forward(self,
                bev_query,
                feats_img,
                feats_pts,
                *args,
                bev_pos=None,
                # imp_pos=None,
                # exp_pos=None,
                spatial_shapes_img=None,
                level_start_index_img=None,
                spatial_shapes_pts=None,
                level_start_index_pts=None,
                bev_h=None,
                bev_w=None,
                ref_2d=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        # intermediate = []

        # bs = gs_xy.shape[0]
        # len_bev = gs_xy.shape[1]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        # bev_query = bev_query.permute(1, 0, 2)
        # bev_pos = bev_pos.permute(1, 0, 2)
        output = bev_query
        intermediate = []

        bs=bev_query.shape[0]
        len_bev=bev_query.shape[1]

        with torch.autocast(device_type='cuda', enabled=False):  # Disable autocasting

            ref_pos = (ref_2d[:, :, None, :2] + 32) / 64
            
            hybird_ref_2d = torch.cat([ref_pos, ref_pos])

            zs = torch.linspace(self.pc_range[2] - self.lidar_height, self.pc_range[5] - self.lidar_height, self.num_points_in_pillar, dtype=torch.float32,
                                device=ref_2d.device)

            zs = zs[None, None, :, None].repeat(bs, len_bev, 1, 1)  # b, N, num_points_in_pillar, 1

            ref_pos = self.compute_corners(bev_query, ref_2d.reshape(-1,3)).reshape(-1,len_bev,4,2)

            zs = zs.repeat(1, 1, self.num_corners + self.num_learnable_pts, 1) # b, N, num_points_in_pillar * (4+0), 1

            ref_3d = torch.cat([ref_pos.repeat(1, 1, self.num_points_in_pillar, 1), zs], dim=-1).permute(2, 0, 1, 3)
            reference_points_cam = ref_3d.to(torch.float32).clone()
            reference_points_cam = torch.cat((reference_points_cam, torch.ones_like(reference_points_cam[..., :1])), -1)    # 4*D, b, N, 4
            reference_points_pts = ref_pos.permute(2, 0, 1, 3)  # 4, b, N, 2

            reference_points_img, bev_mask_img = self.point_sampling(
                reference_points_cam,  kwargs['img_metas']) # num_cam, b, N, 4*D, 2
            reference_points_pts, bev_mask_pts = self.point_sampling_pts(
                reference_points_pts)   # 1, b, N, 5 ,2

        # bev_pos = torch.cat([traj_pos, gs_pos], dim=1)  # bs, len_bev, embed_dim
        # bev_query = query  # bs, len_bev, embed_dim

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                feats_img=feats_img,
                feats_pts=feats_pts,
                *args,
                bev_pos=bev_pos,
                # imp_pos=imp_pos,
                # exp_pos=exp_pos,
                ref_2d=hybird_ref_2d,
                # ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes_img=spatial_shapes_img,
                level_start_index_img=level_start_index_img,
                spatial_shapes_pts=spatial_shapes_pts,
                level_start_index_pts=level_start_index_pts,
                reference_points_img=reference_points_img,
                reference_points_pts=reference_points_pts,
                mask_img=bev_mask_img,
                mask_pts=bev_mask_pts,
                **kwargs
                )

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return bev_query


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        # assert len(operation_order) == 9
        # assert set(operation_order) == set(
        #     ['gs_attn', 'self_attn', 'norm', 'cross_attn_pts', 'cross_attn_img', 'ffn'])

    def forward(self,
                query,
                feats_img=None,
                feats_pts=None,
                query_pos=None,
                key_pos=None,
                imp_pos=None,
                exp_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                bev_pos=None,
                ref_2d=None,
                # ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_img=None,
                mask_img=None,
                reference_points_pts=None,
                mask_pts=None,
                spatial_shapes_img=None,
                level_start_index_img=None,
                spatial_shapes_pts=None,
                level_start_index_pts=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # self attention
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    query_pos=bev_pos,
                    )
                # query = self.attentions[attn_index](
                #     query,
                #     prev_bev,
                #     prev_bev,
                #     identity if self.pre_norm else None,
                #     query_pos=bev_pos,
                #     key_pos=bev_pos,
                #     attn_mask=attn_masks[attn_index],
                #     key_padding_mask=query_key_padding_mask,
                #     reference_points=ref_2d,
                #     spatial_shapes=torch.tensor(
                #         [[bev_h, bev_w]], device=query.device),
                #     level_start_index=torch.tensor([0], device=query.device),
                #     **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn_img':
                query = self.attentions[attn_index](
                    query,
                    feats_img,
                    feats_img,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=reference_points_img,
                    bev_mask=mask_img,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes_img,
                    level_start_index=level_start_index_img,
                    **kwargs)
                attn_index += 1
                identity = query

            # spaital cross attention
            elif layer == 'cross_attn_pts':
                query = self.attentions[attn_index](
                    query,
                    feats_pts,
                    feats_pts,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=reference_points_pts,
                    bev_mask=mask_pts,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes_pts,
                    level_start_index=level_start_index_pts,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoderWithDenoising(BEVFormerEncoder):
    """
    BEVFormerEncoder with noise-injected denoising alignment for LiDAR-camera fusion.
    
    This encoder implements noise decomposition (inj + sys) with improvement margin:
    1. Noise injection to image reference points during training
    2. Noise decomposition: delta_inj (supervised) + delta_sys (unsupervised)
    3. Closed-loop correction and re-sampling
    4. InfoNCE contrastive loss with improvement margin
    
    Args:
        noise_scale: Maximum noise magnitude (dx, dy) in meters for injection
        yaw_scale: Maximum yaw noise in radians
        include_yaw: Whether to include yaw in noise
        sys_scale: Maximum magnitude for systematic noise (default 0.2m)
        proj_channels: Projection dimension for contrastive loss
        denoise_hidden: Hidden dimension for denoiser MLP
        infonce_tau: Temperature for InfoNCE loss
        infonce_radius: Radius for positive samples in meters
        lambda_inj: Weight for injection noise regression loss
        lambda_nce: Weight for InfoNCE loss (post-correction)
        lambda_imp: Weight for improvement margin loss
        margin: Margin for improvement loss
    """
    
    def __init__(
        self,
        *args,
        noise_scale: Tuple[float, float] = (2.0, 2.0),
        yaw_scale: float = 0.1,
        include_yaw: bool = False,
        sys_scale: float = 0.2,
        proj_channels: int = 128,
        denoise_hidden: int = 256,
        infonce_tau: float = 0.1,
        infonce_radius: float = 1.0,
        infonce_topk: Optional[int] = None,
        lambda_noise: float = 1.0,  # Backward compat, maps to lambda_inj
        lambda_inj: float = 1.0,
        lambda_nce: float = 0.1,
        lambda_imp: float = 0.1,
        lambda_reg: float = 0.01,  # Backward compat, deprecated
        margin: float = 0.1,
        **kwargs
    ):
        super(BEVFormerEncoderWithDenoising, self).__init__(*args, **kwargs)
        
        # Denoising alignment module with noise decomposition
        self.denoising_module = DenoisingAlignmentModule(
            embed_dims=self.embed_dims,
            noise_scale=noise_scale,
            yaw_scale=yaw_scale,
            include_yaw=include_yaw,
            sys_scale=sys_scale,
            proj_channels=proj_channels,
            denoise_hidden=denoise_hidden,
            pool_type='mean',
            infonce_tau=infonce_tau,
            infonce_radius=infonce_radius,
            infonce_topk=infonce_topk,
            lambda_noise=lambda_noise,
            lambda_inj=lambda_inj,
            lambda_nce=lambda_nce,
            lambda_imp=lambda_imp,
            lambda_reg=lambda_reg,
            margin=margin,
        )
        
        self.noise_scale = noise_scale
        self.include_yaw = include_yaw
    
    @auto_fp16()
    def forward(
        self,
        bev_query,
        feats_img,
        feats_pts,
        *args,
        bev_pos=None,
        spatial_shapes_img=None,
        level_start_index_img=None,
        spatial_shapes_pts=None,
        level_start_index_pts=None,
        bev_h=None,
        bev_w=None,
        ref_2d=None,
        training: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward function with denoising alignment.
        
        Args:
            bev_query: Input BEV query [B, N, C]
            feats_img: Image features
            feats_pts: Point cloud features
            bev_pos: Positional encoding
            spatial_shapes_img: Spatial shapes for image features
            level_start_index_img: Level start indices for image features
            spatial_shapes_pts: Spatial shapes for point features
            level_start_index_pts: Level start indices for point features
            bev_h: BEV height
            bev_w: BEV width
            ref_2d: Reference 2D positions [B, N, 3] (x, y, heading)
            training: Whether in training mode
            
        Returns:
            output: Final fused query [B, N, C]
            denoising_output: Dictionary containing denoising outputs and losses
        """
        output = bev_query
        intermediate = []
        denoising_outputs = {}
        
        bs = bev_query.shape[0]
        len_bev = bev_query.shape[1]
        device = bev_query.device
        dtype = bev_query.dtype
        
        with torch.autocast(device_type='cuda', enabled=False):
            # Compute reference positions (corners) in BEV coordinates (meters)
            ref_pos = self.compute_corners(bev_query, ref_2d.reshape(-1, 3)).reshape(-1, len_bev, 4, 2)
            # ref_pos: [B, N, 4, 2] in meters
            
            # Sample or set noise
            if training:
                eps_gt = self.denoising_module.sample_noise(bs, device, dtype)
            else:
                # During inference, eps_gt = 0 (no injected noise)
                noise_dim = 3 if self.include_yaw else 2
                eps_gt = torch.zeros(bs, noise_dim, device=device, dtype=dtype)
            
            # Apply noise with adaptive fallback to ensure sufficient valid reference points
            # This prevents all reference points from being pushed outside image bounds
            min_valid_ratio = 0.05  # At least 5% of points should be valid
            max_retries = 3
            ref_pos_noisy = None
            reference_points_img_noisy = None
            bev_mask_img_noisy = None
            
            for retry in range(max_retries):
                # Scale down noise on retries
                noise_scale_factor = 1.0 / (2 ** retry) if retry > 0 else 1.0
                eps_gt_scaled = eps_gt * noise_scale_factor
                
                # Apply noise to get noisy reference positions
                ref_pos_noisy = self.denoising_module.apply_noise_to_ref(ref_pos, eps_gt_scaled)
                
                # Prepare reference points for image cross-attention (noisy)
                zs = torch.linspace(
                    self.pc_range[2] - self.lidar_height,
                    self.pc_range[5] - self.lidar_height,
                    self.num_points_in_pillar,
                    dtype=torch.float32,
                    device=ref_2d.device
                )
                zs = zs[None, None, :, None].repeat(bs, len_bev, 1, 1)
                zs = zs.repeat(1, 1, self.num_corners + self.num_learnable_pts, 1)
                
                # Noisy image reference points
                ref_3d_noisy = torch.cat([
                    ref_pos_noisy.repeat(1, 1, self.num_points_in_pillar, 1), zs
                ], dim=-1).permute(2, 0, 1, 3)
                reference_points_cam_noisy = ref_3d_noisy.to(torch.float32).clone()
                reference_points_cam_noisy = torch.cat(
                    (reference_points_cam_noisy, torch.ones_like(reference_points_cam_noisy[..., :1])), -1
                )
                
                reference_points_img_noisy, bev_mask_img_noisy = self.point_sampling(
                    reference_points_cam_noisy, kwargs['img_metas']
                )
                
                # Check valid point ratio: bev_mask_img_noisy shape is (num_cam, B, N, D)
                # Count valid points per batch
                valid_ratio = bev_mask_img_noisy.float().mean()
                
                # If we have enough valid points, break
                if valid_ratio >= min_valid_ratio:
                    # Update eps_gt to the scaled version that worked
                    if retry > 0:
                        eps_gt = eps_gt_scaled
                    break
                
                # On last retry, use original reference points (no noise) as fallback
                if retry == max_retries - 1:
                    ref_pos_noisy = ref_pos.clone()
                    eps_gt = torch.zeros_like(eps_gt)  # No noise applied
                    # Recompute with original reference points
                    ref_3d_clean = torch.cat([
                        ref_pos_noisy.repeat(1, 1, self.num_points_in_pillar, 1), zs
                    ], dim=-1).permute(2, 0, 1, 3)
                    reference_points_cam_clean = ref_3d_clean.to(torch.float32).clone()
                    reference_points_cam_clean = torch.cat(
                        (reference_points_cam_clean, torch.ones_like(reference_points_cam_clean[..., :1])), -1
                    )
                    reference_points_img_noisy, bev_mask_img_noisy = self.point_sampling(
                        reference_points_cam_clean, kwargs['img_metas']
                    )
            
            # Clean LiDAR reference points (no noise)
            reference_points_pts_clean = ref_pos.permute(2, 0, 1, 3)
            reference_points_pts, bev_mask_pts = self.point_sampling_pts(reference_points_pts_clean)
            
            # Normalized ref_2d for self-attention
            ref_pos_norm = (ref_2d[:, :, None, :2] + 32) / 64
            hybird_ref_2d = torch.cat([ref_pos_norm, ref_pos_norm])
        
        # Process through layers with denoising alignment
        for lid, layer in enumerate(self.layers):
            if isinstance(layer, BEVFormerLayerWithDenoising):
                output, layer_denoising = layer(
                    output,
                    feats_img=feats_img,
                    feats_pts=feats_pts,
                    *args,
                    bev_pos=bev_pos,
                    ref_2d=hybird_ref_2d,
                    bev_h=bev_h,
                    bev_w=bev_w,
                    spatial_shapes_img=spatial_shapes_img,
                    level_start_index_img=level_start_index_img,
                    spatial_shapes_pts=spatial_shapes_pts,
                    level_start_index_pts=level_start_index_pts,
                    reference_points_img_noisy=reference_points_img_noisy,
                    mask_img_noisy=bev_mask_img_noisy,
                    reference_points_pts=reference_points_pts,
                    mask_pts=bev_mask_pts,
                    ref_pos=ref_pos,
                    ref_pos_noisy=ref_pos_noisy,
                    eps_gt=eps_gt,
                    denoising_module=self.denoising_module,
                    point_sampling_fn=self.point_sampling,
                    pc_range=self.pc_range,
                    lidar_height=self.lidar_height,
                    num_points_in_pillar=self.num_points_in_pillar,
                    num_corners=self.num_corners,
                    num_learnable_pts=self.num_learnable_pts,
                    training=training,
                    **kwargs
                )
                
                # Accumulate denoising outputs
                for key, value in layer_denoising.items():
                    if key not in denoising_outputs:
                        denoising_outputs[key] = []
                    denoising_outputs[key].append(value)
            else:
                # Fallback to standard layer
                # Compute clean image reference points for standard layer
                ref_3d_clean = torch.cat([
                    ref_pos.repeat(1, 1, self.num_points_in_pillar, 1), zs
                ], dim=-1).permute(2, 0, 1, 3)
                reference_points_cam_clean = ref_3d_clean.to(torch.float32).clone()
                reference_points_cam_clean = torch.cat(
                    (reference_points_cam_clean, torch.ones_like(reference_points_cam_clean[..., :1])), -1
                )
                reference_points_img_clean, bev_mask_img_clean = self.point_sampling(
                    reference_points_cam_clean, kwargs['img_metas']
                )
                
                output = layer(
                    output,
                    feats_img=feats_img,
                    feats_pts=feats_pts,
                    *args,
                    bev_pos=bev_pos,
                    ref_2d=hybird_ref_2d,
                    bev_h=bev_h,
                    bev_w=bev_w,
                    spatial_shapes_img=spatial_shapes_img,
                    level_start_index_img=level_start_index_img,
                    spatial_shapes_pts=spatial_shapes_pts,
                    level_start_index_pts=level_start_index_pts,
                    reference_points_img=reference_points_img_clean,
                    mask_img=bev_mask_img_clean,
                    reference_points_pts=reference_points_pts,
                    mask_pts=bev_mask_pts,
                    **kwargs
                )
            
            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)
        
        # Aggregate losses across layers
        aggregated_losses = {}
        for key, values in denoising_outputs.items():
            if key.startswith('loss_'):
                # Average losses across layers
                aggregated_losses[key] = torch.stack(values).mean()
            else:
                # Keep last layer's output for non-loss items
                aggregated_losses[key] = values[-1]
        
        # Add eps_gt to output
        aggregated_losses['eps_gt'] = eps_gt
        
        if self.return_intermediate:
            return torch.stack(intermediate), aggregated_losses
        
        return output, aggregated_losses


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayerWithDenoising(BEVFormerLayer):
    """
    BEVFormerLayer with parallel cross-attention and denoising alignment.
    
    This layer implements noise decomposition (inj + sys) with improvement margin:
    1. Q0 = self_attn(query, pos_encoding) + query (residual)
    2. Q0 = norm(Q0)
    3. Parallel branches from Q0:
       - lidar_query = cross_attn_pts(Q0, feats_pts) + Q0 (residual)
       - lidar_query = norm(lidar_query)
       - img_query_noisy = cross_attn_img(Q0, feats_img, ref_pos_noisy) + Q0 (residual)
       - img_query_noisy = norm(img_query_noisy)
    4. Noise decomposition:
       - delta_inj, delta_sys, delta_pred = DenoiseHead(lidar_query - img_query_noisy)
       - delta_inj: supervised by eps_gt, range ±inj_scale
       - delta_sys: NOT supervised, ||delta_sys|| ∈ [0, sys_scale]
    5. ref_pos_corr = ApplyNoise(ref_pos, eps_gt - delta_pred)  [train]
       ref_pos_corr = ApplyNoise(ref_pos, -delta_pred)          [inference, eps_gt=0]
    6. img_query_corr = cross_attn_img(lidar_query, feats_img, ref_pos_corr) + lidar_query (residual)
    7. img_query_corr = norm(img_query_corr)
    8. fuse_query = lidar_query + img_query_corr
    9. output = ffn(fuse_query) + fuse_query (residual)
    10. output = norm(output)
    
    Losses (training):
    - L_inj = SmoothL1(delta_inj, eps_gt)  [only delta_inj supervised]
    - L_nce_pre = InfoNCE(lidar + img_noisy, lidar)
    - L_nce_post = InfoNCE(lidar + img_corr, lidar)
    - L_imp = relu(margin + L_nce_post - L_nce_pre)
    """
    
    def __init__(self, *args, **kwargs):
        super(BEVFormerLayerWithDenoising, self).__init__(*args, **kwargs)
    
    def forward(
        self,
        query,
        feats_img=None,
        feats_pts=None,
        query_pos=None,
        key_pos=None,
        imp_pos=None,
        exp_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        bev_pos=None,
        ref_2d=None,
        bev_h=None,
        bev_w=None,
        reference_points_img_noisy=None,
        mask_img_noisy=None,
        reference_points_pts=None,
        mask_pts=None,
        spatial_shapes_img=None,
        level_start_index_img=None,
        spatial_shapes_pts=None,
        level_start_index_pts=None,
        prev_bev=None,
        ref_pos=None,
        ref_pos_noisy=None,
        eps_gt=None,
        denoising_module=None,
        point_sampling_fn=None,
        pc_range=None,
        lidar_height=None,
        num_points_in_pillar=None,
        num_corners=None,
        num_learnable_pts=None,
        training=True,
        **kwargs
    ):
        """
        Forward function with denoising alignment.
        
        The flow matches original BEVFormerLayer structure:
        1. self_attn -> norm
        2. Parallel: cross_attn_pts -> norm, cross_attn_img(noisy) -> norm
        3. Denoise prediction and correction
        4. cross_attn_img(corrected) -> norm
        5. Fusion -> ffn -> norm
        """
        denoising_output = {}
        
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
        
        # Track which attention modules we've used
        self_attn_module = None
        pts_attn_module = None
        img_attn_module = None
        
        # Find attention modules by operation order
        # Original order: ('self_attn', 'norm', 'cross_attn_pts', 'norm', 'cross_attn_img', 'norm', 'ffn', 'norm')
        temp_attn_index = 0
        for layer in self.operation_order:
            if layer == 'self_attn':
                self_attn_module = self.attentions[temp_attn_index]
                temp_attn_index += 1
            elif layer == 'cross_attn_pts':
                pts_attn_module = self.attentions[temp_attn_index]
                temp_attn_index += 1
            elif layer == 'cross_attn_img':
                img_attn_module = self.attentions[temp_attn_index]
                temp_attn_index += 1
        
        # ============================================================
        # Step 1: Self-attention (with residual inside SelfAttention)
        # ============================================================
        Q0 = self_attn_module(query, query_pos=bev_pos)
        # Step 2: Norm after self-attention
        Q0 = self.norms[0](Q0)
        
        # ============================================================
        # Step 3: Parallel cross-attention branches
        # ============================================================
        
        # 3.1 LiDAR cross-attention (clean, no noise)
        # SpatialCrossAttention has internal residual: output = dropout(slots) + inp_residual
        lidar_query = pts_attn_module(
            Q0,
            feats_pts,
            feats_pts,
            None,  # residual=None means use query as residual internally
            query_pos=query_pos,
            key_pos=key_pos,
            reference_points=reference_points_pts,
            bev_mask=mask_pts,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes_pts,
            level_start_index=level_start_index_pts,
            **kwargs
        )
        # Norm after cross_attn_pts
        lidar_query = self.norms[1](lidar_query)
        
        # 3.2 Image cross-attention with noisy reference points
        img_query_noisy = img_attn_module(
            Q0,
            feats_img,
            feats_img,
            None,  # residual=None means use query as residual internally
            query_pos=query_pos,
            key_pos=key_pos,
            reference_points=reference_points_img_noisy,
            bev_mask=mask_img_noisy,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes_img,
            level_start_index=level_start_index_img,
            **kwargs
        )
        # Norm after cross_attn_img (noisy)
        img_query_noisy = self.norms[2](img_query_noisy)
        
        # ============================================================
        # Step 4: Predict noise decomposition from feature difference
        # delta_inj: supervised by eps_gt, range ±inj_scale
        # delta_sys: NOT supervised, ||delta_sys|| ∈ [0, sys_scale]
        # delta_pred = delta_inj + delta_sys
        # ============================================================
        delta_inj, delta_sys, delta_pred = denoising_module.predict_noise_decomposed(
            lidar_query, img_query_noisy
        )
        
        # ============================================================
        # Step 5: Compute corrected reference positions
        # Training: ref_pos_corr = ApplyNoise(ref_pos, eps_gt - delta_pred)
        # Inference: ref_pos_corr = ApplyNoise(ref_pos, -delta_pred) since eps_gt=0
        # ============================================================
        residual_noise = eps_gt - delta_pred
        ref_pos_corr = denoising_module.apply_noise_to_ref(ref_pos, residual_noise)
        
        # Compute corrected image reference points
        bs, len_bev = query.shape[:2]
        device = query.device
        
        with torch.autocast(device_type='cuda', enabled=False):
            zs = torch.linspace(
                pc_range[2] - lidar_height,
                pc_range[5] - lidar_height,
                num_points_in_pillar,
                dtype=torch.float32,
                device=device
            )
            zs = zs[None, None, :, None].repeat(bs, len_bev, 1, 1)
            zs = zs.repeat(1, 1, num_corners + num_learnable_pts, 1)
            
            ref_3d_corr = torch.cat([
                ref_pos_corr.repeat(1, 1, num_points_in_pillar, 1), zs
            ], dim=-1).permute(2, 0, 1, 3)
            reference_points_cam_corr = ref_3d_corr.to(torch.float32).clone()
            reference_points_cam_corr = torch.cat(
                (reference_points_cam_corr, torch.ones_like(reference_points_cam_corr[..., :1])), -1
            )
            
            reference_points_img_corr, bev_mask_img_corr = point_sampling_fn(
                reference_points_cam_corr, kwargs['img_metas']
            )
        
        # ============================================================
        # Step 6: Corrected image cross-attention
        # ============================================================
        # Use lidar_query as input for second pass (corrected sampling)
        img_query_corr = img_attn_module(
            lidar_query,
            feats_img,
            feats_img,
            None,  # residual=None means use query (lidar_query) as residual internally
            query_pos=query_pos,
            key_pos=key_pos,
            reference_points=reference_points_img_corr,
            bev_mask=bev_mask_img_corr,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes_img,
            level_start_index=level_start_index_img,
            **kwargs
        )
        # Step 7: Norm after corrected cross_attn_img
        img_query_corr = self.norms[3](img_query_corr)
        
        # ============================================================
        # Step 8: Fuse queries
        # ============================================================
        fuse_query = lidar_query + img_query_corr
        
        # ============================================================
        # Step 9: FFN with residual
        # ============================================================
        if len(self.ffns) > 0:
            # FFN has internal residual: output = ffn(x) + identity
            output = self.ffns[0](fuse_query, fuse_query)
        else:
            output = fuse_query
        
        # Step 10: Final norm after FFN
        if len(self.norms) > 4:
            output = self.norms[4](output)
        
        # ============================================================
        # Compute losses (training only)
        # Uses noise decomposition and improvement margin loss
        # ============================================================
        if training:
            # Get 2D positions for InfoNCE
            ref_pos_2d = ref_pos.mean(dim=2) if ref_pos.dim() == 4 else ref_pos
            
            losses = denoising_module.compute_losses(
                delta_inj=delta_inj,
                delta_sys=delta_sys,
                delta_pred=delta_pred,
                eps_gt=eps_gt,
                fuse_query=fuse_query,
                lidar_query=lidar_query,
                img_query_noisy=img_query_noisy,
                ref_pos=ref_pos_2d,
                valid_mask=None,
            )
            denoising_output.update(losses)
        
        # Store outputs for debugging/visualization
        denoising_output['delta_inj'] = delta_inj
        denoising_output['delta_sys'] = delta_sys
        denoising_output['delta_pred'] = delta_pred
        denoising_output['noise_pred'] = delta_pred  # Backward compatibility
        denoising_output['ref_pos_noisy'] = ref_pos_noisy
        denoising_output['ref_pos_corr'] = ref_pos_corr
        denoising_output['lidar_query'] = lidar_query
        denoising_output['img_query_noisy'] = img_query_noisy
        denoising_output['img_query_corr'] = img_query_corr
        denoising_output['fuse_query'] = fuse_query
        
        return output, denoising_output
