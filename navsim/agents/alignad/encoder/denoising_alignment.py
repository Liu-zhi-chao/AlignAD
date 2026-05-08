"""
Noise-Injected Denoising Alignment Module for LiDAR-Camera Misalignment Correction.

This module implements a closed-loop calibration mechanism that:
1. Injects global SE(2) noise to image cross-attention reference points during training
2. Predicts the injected noise using a denoiser MLP
3. Corrects the reference points and re-samples image features
4. Uses InfoNCE contrastive loss for alignment consistency
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, NamedTuple


class DenoisingAlignmentOutput(NamedTuple):
    """Output container for denoising alignment module."""
    fuse_query: torch.Tensor           # [B, N, C] - Final fused query
    lidar_query: torch.Tensor          # [B, N, C] - LiDAR cross-attention output
    img_query_noisy: torch.Tensor      # [B, N, C] - Image cross-attention with noisy ref
    img_query_corr: torch.Tensor       # [B, N, C] - Image cross-attention with corrected ref
    noise_pred: torch.Tensor           # [B, 1, 4, 2] or [B, 3] - Predicted noise
    eps_gt: torch.Tensor               # [B, 1, 4, 2] or [B, 3] - Ground truth noise (training)
    ref_pos_noisy: torch.Tensor        # [B, N, 4, 2] - Noisy reference positions
    ref_pos_corr: torch.Tensor         # [B, N, 4, 2] - Corrected reference positions


class ApplyGlobalNoise(nn.Module):
    """
    Apply global SE(2) transformation (dx, dy, dyaw) to BEV reference positions.
    
    The transformation is applied as a rigid body transformation in BEV space:
    - Translation: (dx, dy) in meters
    - Rotation: dyaw in radians (optional)
    
    Args:
        noise_scale: Maximum noise magnitude for (dx, dy) in meters. Default: (2.0, 2.0)
        yaw_scale: Maximum yaw noise in radians. Default: 0.1
        include_yaw: Whether to include yaw rotation. Default: False
    """
    
    def __init__(
        self,
        noise_scale: Tuple[float, float] = (2.0, 2.0),
        yaw_scale: float = 0.1,
        include_yaw: bool = False,
    ):
        super().__init__()
        self.noise_scale = noise_scale
        self.yaw_scale = yaw_scale
        self.include_yaw = include_yaw
        
    def forward(
        self,
        ref_pos: torch.Tensor,
        eps: torch.Tensor,
        center: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply global noise transformation to reference positions.
        
        Args:
            ref_pos: Reference positions in BEV coordinates [B, N, num_corners, 2] or [B, N, 2]
                     Units are in meters.
            eps: Noise to apply. Shape depends on include_yaw:
                 - If include_yaw=False: [B, 2] for (dx, dy)
                 - If include_yaw=True: [B, 3] for (dx, dy, dyaw)
                 Or [B, 1, 4, 2] for per-corner noise (broadcast across N)
            center: Optional rotation center [B, 2]. If None, uses origin (0, 0).
        
        Returns:
            Transformed reference positions with same shape as input.
        """
        original_shape = ref_pos.shape
        batch_size = ref_pos.shape[0]
        device = ref_pos.device
        dtype = ref_pos.dtype
        
        # Handle different eps shapes
        if eps.dim() == 4:
            # eps shape: [B, 1, 4, 2] - per-corner translation noise
            # Broadcast to [B, N, 4, 2]
            if ref_pos.dim() == 4:
                # ref_pos: [B, N, 4, 2]
                dx = eps[..., 0]  # [B, 1, 4]
                dy = eps[..., 1]  # [B, 1, 4]
                ref_pos_transformed = ref_pos.clone()
                ref_pos_transformed[..., 0] = ref_pos[..., 0] + dx
                ref_pos_transformed[..., 1] = ref_pos[..., 1] + dy
            else:
                raise ValueError(f"ref_pos shape {ref_pos.shape} incompatible with eps shape {eps.shape}")
        elif eps.dim() == 2:
            # eps shape: [B, 2] or [B, 3]
            if eps.shape[-1] == 2:
                # Translation only
                dx, dy = eps[:, 0], eps[:, 1]
                
                if ref_pos.dim() == 4:
                    # ref_pos: [B, N, 4, 2]
                    ref_pos_transformed = ref_pos.clone()
                    ref_pos_transformed[..., 0] = ref_pos[..., 0] + dx[:, None, None]
                    ref_pos_transformed[..., 1] = ref_pos[..., 1] + dy[:, None, None]
                elif ref_pos.dim() == 3:
                    # ref_pos: [B, N, 2]
                    ref_pos_transformed = ref_pos.clone()
                    ref_pos_transformed[..., 0] = ref_pos[..., 0] + dx[:, None]
                    ref_pos_transformed[..., 1] = ref_pos[..., 1] + dy[:, None]
                else:
                    raise ValueError(f"Unsupported ref_pos shape: {ref_pos.shape}")
                    
            elif eps.shape[-1] == 3 and self.include_yaw:
                # Translation + rotation
                dx, dy, dyaw = eps[:, 0], eps[:, 1], eps[:, 2]
                
                # Rotation matrix
                cos_yaw = torch.cos(dyaw)
                sin_yaw = torch.sin(dyaw)
                
                if center is None:
                    center = torch.zeros(batch_size, 2, device=device, dtype=dtype)
                
                if ref_pos.dim() == 4:
                    # ref_pos: [B, N, 4, 2]
                    # Center the points
                    centered = ref_pos - center[:, None, None, :]
                    
                    # Apply rotation
                    rotated_x = centered[..., 0] * cos_yaw[:, None, None] - centered[..., 1] * sin_yaw[:, None, None]
                    rotated_y = centered[..., 0] * sin_yaw[:, None, None] + centered[..., 1] * cos_yaw[:, None, None]
                    
                    # Apply translation and restore center
                    ref_pos_transformed = torch.stack([
                        rotated_x + center[:, None, None, 0] + dx[:, None, None],
                        rotated_y + center[:, None, None, 1] + dy[:, None, None]
                    ], dim=-1)
                elif ref_pos.dim() == 3:
                    # ref_pos: [B, N, 2]
                    centered = ref_pos - center[:, None, :]
                    
                    rotated_x = centered[..., 0] * cos_yaw[:, None] - centered[..., 1] * sin_yaw[:, None]
                    rotated_y = centered[..., 0] * sin_yaw[:, None] + centered[..., 1] * cos_yaw[:, None]
                    
                    ref_pos_transformed = torch.stack([
                        rotated_x + center[:, None, 0] + dx[:, None],
                        rotated_y + center[:, None, 1] + dy[:, None]
                    ], dim=-1)
                else:
                    raise ValueError(f"Unsupported ref_pos shape: {ref_pos.shape}")
            else:
                raise ValueError(f"Invalid eps shape: {eps.shape}")
        else:
            raise ValueError(f"Unsupported eps dimension: {eps.dim()}")
        
        assert ref_pos_transformed.shape == original_shape, \
            f"Shape mismatch: {ref_pos_transformed.shape} vs {original_shape}"
        
        return ref_pos_transformed
    
    def sample_noise(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Sample random global noise for training.
        
        Args:
            batch_size: Number of samples in batch
            device: Target device
            dtype: Target dtype
            
        Returns:
            Sampled noise tensor:
            - Shape [B, 2] if include_yaw=False
            - Shape [B, 3] if include_yaw=True
        """
        # Sample uniform noise in [-scale, scale]
        dx = (torch.rand(batch_size, device=device, dtype=dtype) * 2 - 1) * self.noise_scale[0]
        dy = (torch.rand(batch_size, device=device, dtype=dtype) * 2 - 1) * self.noise_scale[1]
        
        if self.include_yaw:
            dyaw = (torch.rand(batch_size, device=device, dtype=dtype) * 2 - 1) * self.yaw_scale
            return torch.stack([dx, dy, dyaw], dim=-1)
        else:
            return torch.stack([dx, dy], dim=-1)


class DenoiseHead(nn.Module):
    """
    MLP head for predicting noise decomposition: delta_inj (injected) + delta_sys (systematic).
    
    Takes the difference between LiDAR and noisy image features and predicts:
    - delta_inj: Explains injected noise, supervised by eps_gt, range ±inj_scale (default ±2m)
    - delta_sys: Explains systematic error, NOT supervised by eps_gt, range [0, sys_scale] (default 0~0.2m)
    
    Args:
        in_channels: Input feature dimension (C)
        hidden_channels: Hidden layer dimension
        noise_dim: Output noise dimension (2 for dx,dy)
        inj_scale: Scale for delta_inj (max magnitude, default 2.0m)
        sys_scale: Scale for delta_sys (max magnitude, default 0.2m)
        pool_type: How to pool spatial features ('mean', 'max', 'attention')
        num_layers: Number of MLP layers for shared backbone
    """
    
    def __init__(
        self,
        in_channels: int = 512,
        hidden_channels: int = 256,
        noise_dim: int = 2,
        noise_scale: Tuple[float, ...] = (2.0, 2.0),  # For backward compatibility
        inj_scale: Tuple[float, float] = (2.0, 2.0),
        sys_scale: float = 0.2,
        pool_type: str = 'mean',
        num_layers: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.noise_dim = noise_dim
        self.pool_type = pool_type
        self.sys_scale = sys_scale
        
        # Use inj_scale if provided, otherwise fall back to noise_scale
        self.inj_scale = nn.Parameter(
            torch.tensor(inj_scale if inj_scale != (2.0, 2.0) else noise_scale, dtype=torch.float32),
            requires_grad=False
        )
        
        # Layer normalization for input
        self.input_norm = nn.LayerNorm(in_channels)
        
        # Build shared MLP backbone
        backbone_layers = []
        curr_dim = in_channels
        for i in range(num_layers - 1):
            backbone_layers.extend([
                nn.Linear(curr_dim, hidden_channels),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_channels),
            ])
            curr_dim = hidden_channels
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # Injection head: outputs delta_inj [B, 2], clamped by tanh * inj_scale
        self.inj_head = nn.Linear(hidden_channels, noise_dim)
        
        # Systematic head: outputs direction [B, 2] and amplitude [B, 1]
        # delta_sys = normalize(direction) * sigmoid(amplitude) * sys_scale
        # This ensures ||delta_sys|| ∈ [0, sys_scale]
        self.sys_dir_head = nn.Linear(hidden_channels, noise_dim)  # Direction
        self.sys_amp_head = nn.Linear(hidden_channels, 1)  # Amplitude
        
        # Attention pooling (optional)
        if pool_type == 'attention':
            self.attn_pool = nn.Sequential(
                nn.Linear(in_channels, 1),
                nn.Softmax(dim=1)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize sys_amp_head bias to produce small initial amplitude
        # sigmoid(-2) ≈ 0.12, so initial amplitude ≈ 0.12 * 0.2 = 0.024m
        nn.init.constant_(self.sys_amp_head.bias, -2.0)
    
    def forward(
        self,
        lidar_query: torch.Tensor,
        img_query_noisy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict noise decomposition from feature difference.
        
        Args:
            lidar_query: LiDAR cross-attention output [B, N, C]
            img_query_noisy: Noisy image cross-attention output [B, N, C]
            
        Returns:
            delta_inj: Injected noise prediction [B, 2], range ±inj_scale
            delta_sys: Systematic noise prediction [B, 2], ||delta_sys|| ∈ [0, sys_scale]
            delta_pred: Total prediction delta_inj + delta_sys [B, 2]
        """
        # Compute feature difference
        diff = lidar_query - img_query_noisy  # [B, N, C]
        
        # Apply layer normalization
        diff = self.input_norm(diff)
        
        # Pool across spatial dimension
        if self.pool_type == 'mean':
            pooled = diff.mean(dim=1)  # [B, C]
        elif self.pool_type == 'max':
            pooled = diff.max(dim=1)[0]  # [B, C]
        elif self.pool_type == 'attention':
            attn_weights = self.attn_pool(diff)  # [B, N, 1]
            pooled = (diff * attn_weights).sum(dim=1)  # [B, C]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        # Shared backbone
        features = self.backbone(pooled)  # [B, hidden_channels]
        
        # ============================================================
        # Injection head: delta_inj = tanh(raw) * inj_scale
        # ============================================================
        raw_inj = self.inj_head(features)  # [B, 2]
        delta_inj = torch.tanh(raw_inj) * self.inj_scale  # [B, 2], range ±inj_scale
        
        # ============================================================
        # Systematic head: delta_sys with ||delta_sys|| ∈ [0, sys_scale]
        # Using direction + amplitude formulation
        # ============================================================
        raw_dir = self.sys_dir_head(features)  # [B, 2]
        raw_amp = self.sys_amp_head(features)  # [B, 1]
        
        # Safe normalize direction (add eps to avoid division by zero)
        dir_norm = raw_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = raw_dir / dir_norm  # [B, 2], unit vector
        
        # Amplitude in [0, sys_scale]
        amplitude = torch.sigmoid(raw_amp) * self.sys_scale  # [B, 1]
        
        # delta_sys = direction * amplitude
        delta_sys = direction * amplitude  # [B, 2], ||delta_sys|| ∈ [0, sys_scale]
        
        # Total prediction
        delta_pred = delta_inj + delta_sys  # [B, 2]
        
        return delta_inj, delta_sys, delta_pred
    
    def predict_noise(
        self,
        lidar_query: torch.Tensor,
        img_query_noisy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Backward compatible interface that returns only delta_pred.
        
        For new code, use forward() to get all three outputs.
        """
        _, _, delta_pred = self.forward(lidar_query, img_query_noisy)
        return delta_pred


class ContrastiveProjector(nn.Module):
    """
    Projector for InfoNCE contrastive learning.
    
    Projects high-dimensional features to a lower-dimensional space
    for computing contrastive loss.
    
    Args:
        in_channels: Input feature dimension
        proj_channels: Output projection dimension
        hidden_channels: Hidden layer dimension (if None, uses proj_channels)
        num_layers: Number of projection layers (1 or 2)
    """
    
    def __init__(
        self,
        in_channels: int = 512,
        proj_channels: int = 128,
        hidden_channels: Optional[int] = None,
        num_layers: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.proj_channels = proj_channels
        
        if hidden_channels is None:
            hidden_channels = proj_channels
        
        if num_layers == 1:
            self.projector = nn.Linear(in_channels, proj_channels)
        elif num_layers == 2:
            self.projector = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, proj_channels),
            )
        else:
            raise ValueError(f"num_layers must be 1 or 2, got {num_layers}")
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features and L2 normalize.
        
        Args:
            x: Input features [B, N, C]
            
        Returns:
            Normalized projections [B, N, proj_channels]
        """
        proj = self.projector(x)  # [B, N, proj_channels]
        proj = F.normalize(proj, p=2, dim=-1)  # L2 normalize
        return proj


class DenoisingAlignmentModule(nn.Module):
    """
    Complete denoising alignment module for LiDAR-camera fusion.
    
    This module implements the noise-injected denoising alignment mechanism with
    noise decomposition (inj + sys) and improvement margin loss:
    
    1. Apply global noise to image reference points
    2. Compute parallel cross-attention (image noisy + lidar)
    3. Predict noise decomposition: delta_inj (supervised) + delta_sys (unsupervised)
    4. Re-sample image features with corrected reference points
    5. Fuse the results
    6. Compute losses including improvement margin
    
    Args:
        embed_dims: Feature embedding dimension
        noise_scale: Maximum noise magnitude (dx, dy) in meters for injection
        yaw_scale: Maximum yaw noise in radians
        include_yaw: Whether to include yaw in noise
        sys_scale: Maximum magnitude for systematic noise (default 0.2m)
        proj_channels: Projection dimension for contrastive loss
        denoise_hidden: Hidden dimension for denoiser MLP
        denoise_layers: Number of layers in denoiser MLP
        pool_type: Pooling type for denoiser ('mean', 'max', 'attention')
        infonce_tau: Temperature for InfoNCE loss
        infonce_radius: Radius for positive samples in meters
        infonce_topk: Top-K nearest neighbors as positives (alternative to radius)
        lambda_inj: Weight for injection noise regression loss (delta_inj vs eps_gt)
        lambda_nce: Weight for InfoNCE loss (post-correction)
        lambda_imp: Weight for improvement margin loss
        margin: Margin for improvement loss (L_nce_post should be < L_nce_pre - margin)
    """
    
    def __init__(
        self,
        embed_dims: int = 512,
        noise_scale: Tuple[float, float] = (2.0, 2.0),
        yaw_scale: float = 0.1,
        include_yaw: bool = False,
        sys_scale: float = 0.2,
        proj_channels: int = 128,
        denoise_hidden: int = 256,
        denoise_layers: int = 3,
        pool_type: str = 'mean',
        infonce_tau: float = 0.1,
        infonce_radius: float = 1.0,
        infonce_topk: Optional[int] = None,
        lambda_noise: float = 1.0,  # Backward compat, maps to lambda_inj
        lambda_inj: float = 1.0,
        lambda_nce: float = 0.1,
        lambda_imp: float = 0.1,
        lambda_reg: float = 0.01,  # Kept for backward compat, not used in new loss
        margin: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.include_yaw = include_yaw
        self.sys_scale = sys_scale
        self.infonce_tau = infonce_tau
        self.infonce_radius = infonce_radius
        self.infonce_topk = infonce_topk
        # Use lambda_inj if explicitly set, otherwise fall back to lambda_noise
        self.lambda_inj = lambda_inj if lambda_inj != 1.0 else lambda_noise
        self.lambda_nce = lambda_nce
        self.lambda_imp = lambda_imp
        self.lambda_reg = lambda_reg  # Kept for backward compatibility
        self.margin = margin
        
        noise_dim = 3 if include_yaw else 2
        
        # Noise application module
        self.apply_noise = ApplyGlobalNoise(
            noise_scale=noise_scale,
            yaw_scale=yaw_scale,
            include_yaw=include_yaw,
        )
        
        # Denoiser head with decomposition (inj + sys)
        self.denoise_head = DenoiseHead(
            in_channels=embed_dims,
            hidden_channels=denoise_hidden,
            noise_dim=noise_dim,
            noise_scale=noise_scale if not include_yaw else (*noise_scale, yaw_scale),
            inj_scale=noise_scale[:2] if not include_yaw else noise_scale,
            sys_scale=sys_scale,
            pool_type=pool_type,
            num_layers=denoise_layers,
        )
        
        # Contrastive projector
        self.projector = ContrastiveProjector(
            in_channels=embed_dims,
            proj_channels=proj_channels,
            num_layers=2,
        )
    
    def sample_noise(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample random noise for training."""
        return self.apply_noise.sample_noise(batch_size, device, dtype)
    
    def apply_noise_to_ref(
        self,
        ref_pos: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """Apply noise to reference positions."""
        return self.apply_noise(ref_pos, eps)
    
    def predict_noise(
        self,
        lidar_query: torch.Tensor,
        img_query_noisy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise from feature difference (backward compatible interface).
        Returns only delta_pred = delta_inj + delta_sys.
        """
        return self.denoise_head.predict_noise(lidar_query, img_query_noisy)
    
    def predict_noise_decomposed(
        self,
        lidar_query: torch.Tensor,
        img_query_noisy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict noise decomposition from feature difference.
        
        Returns:
            delta_inj: Injected noise prediction [B, 2], supervised by eps_gt
            delta_sys: Systematic noise prediction [B, 2], ||delta_sys|| ∈ [0, sys_scale]
            delta_pred: Total prediction delta_inj + delta_sys [B, 2]
        """
        return self.denoise_head(lidar_query, img_query_noisy)
    
    def compute_corrected_ref(
        self,
        ref_pos: torch.Tensor,
        eps_gt: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute corrected reference positions.
        
        For training: ref_pos_corr = ApplyNoise(ref_pos, eps_gt - noise_pred)
        For inference: ref_pos_corr = ApplyNoise(ref_pos, -noise_pred) since eps_gt=0
        """
        residual_noise = eps_gt - noise_pred
        return self.apply_noise(ref_pos, residual_noise)
    
    def compute_losses(
        self,
        delta_inj: torch.Tensor,
        delta_sys: torch.Tensor,
        delta_pred: torch.Tensor,
        eps_gt: torch.Tensor,
        fuse_query: torch.Tensor,
        lidar_query: torch.Tensor,
        img_query_noisy: torch.Tensor,
        ref_pos: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all alignment losses with noise decomposition and improvement margin.
        
        Args:
            delta_inj: Predicted injection noise [B, 2], supervised by eps_gt
            delta_sys: Predicted systematic noise [B, 2], NOT supervised
            delta_pred: Total prediction delta_inj + delta_sys [B, 2]
            eps_gt: Ground truth injected noise [B, 2]
            fuse_query: Post-correction fused query (lidar + img_corr) [B, N, C]
            lidar_query: LiDAR query features [B, N, C]
            img_query_noisy: Noisy image query features [B, N, C]
            ref_pos: Reference positions [B, N, 2] or [B, N, 4, 2]
            valid_mask: Optional validity mask [B, N]
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        device = delta_inj.device
        
        # ============================================================
        # 1. Injection noise regression loss (only supervise delta_inj)
        # L_inj = SmoothL1(delta_inj, eps_gt)
        # ============================================================
        loss_inj = F.smooth_l1_loss(delta_inj, eps_gt)
        losses['loss_inj'] = self.lambda_inj * loss_inj
        
        # ============================================================
        # 2. Construct pre-correction and post-correction fused features
        # fuse_pre  = lidar_query + img_query_noisy (simple addition, no extra cross-attn)
        # fuse_post = fuse_query (already computed: lidar_query + img_query_corr)
        # ============================================================
        fuse_pre = lidar_query + img_query_noisy  # [B, N, C]
        fuse_post = fuse_query  # [B, N, C]
        
        # ============================================================
        # 3. InfoNCE losses for pre and post correction
        # ============================================================
        loss_nce_pre = self._compute_infonce_loss(
            fuse_pre, lidar_query, ref_pos, valid_mask
        )
        loss_nce_post = self._compute_infonce_loss(
            fuse_post, lidar_query, ref_pos, valid_mask
        )
        
        losses['loss_nce_pre'] = loss_nce_pre  # For monitoring
        losses['loss_nce_post'] = self.lambda_nce * loss_nce_post
        losses['loss_nce'] = losses['loss_nce_post']  # Backward compatibility
        
        # ============================================================
        # 4. Improvement margin loss
        # L_imp = relu(margin + L_nce_post - L_nce_pre)
        # Meaning: post-correction should be better (lower) than pre-correction by margin
        # ============================================================
        loss_imp = F.relu(self.margin + loss_nce_post - loss_nce_pre)
        losses['loss_imp'] = self.lambda_imp * loss_imp
        
        # ============================================================
        # 5. Systematic noise magnitude monitoring (not a loss, just for logging)
        # ============================================================
        sys_magnitude = delta_sys.norm(dim=-1).mean()
        losses['sys_magnitude'] = sys_magnitude  # For monitoring, not backprop
        
        # ============================================================
        # Total alignment loss
        # L_total = λ_inj * L_inj + λ_nce * L_nce_post + λ_imp * L_imp
        # ============================================================
        losses['loss_alignment'] = (
            losses['loss_inj'] + 
            losses['loss_nce_post'] + 
            losses['loss_imp']
        )
        
        # Backward compatibility: keep loss_noise pointing to loss_inj
        losses['loss_noise'] = losses['loss_inj']
        losses['loss_noise_reg'] = torch.tensor(0.0, device=device)  # Deprecated
        
        return losses
    
    def _compute_infonce_loss(
        self,
        fuse_query: torch.Tensor,
        lidar_query: torch.Tensor,
        ref_pos: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss with local positives based on BEV distance.
        
        Args:
            fuse_query: Anchor features [B, N, C]
            lidar_query: Key features [B, N, C]
            ref_pos: Reference positions for computing positives [B, N, 2] or [B, N, 4, 2]
            valid_mask: Optional validity mask [B, N]
            
        Returns:
            InfoNCE loss scalar
        """
        B, N, C = fuse_query.shape
        device = fuse_query.device
        
        # Project features
        z_anchor = self.projector(fuse_query)  # [B, N, D]
        z_key = self.projector(lidar_query)    # [B, N, D]
        
        # Get 2D positions for distance computation
        if ref_pos.dim() == 4:
            # [B, N, 4, 2] -> [B, N, 2] (use center or first corner)
            pos_2d = ref_pos.mean(dim=2)  # Average of corners
        else:
            pos_2d = ref_pos  # [B, N, 2]
        
        # Compute pairwise distances within each batch
        # pos_2d: [B, N, 2]
        dist_matrix = torch.cdist(pos_2d, pos_2d, p=2)  # [B, N, N]
        
        # Determine positive mask based on distance
        if self.infonce_topk is not None:
            # Top-K nearest neighbors as positives
            _, topk_indices = torch.topk(dist_matrix, k=self.infonce_topk + 1, dim=-1, largest=False)
            positive_mask = torch.zeros(B, N, N, device=device, dtype=torch.bool)
            positive_mask.scatter_(2, topk_indices, True)
            # Exclude self
            positive_mask.diagonal(dim1=1, dim2=2).fill_(False)
        else:
            # Radius-based positives
            positive_mask = (dist_matrix <= self.infonce_radius) & (dist_matrix > 0)
        
        # Compute logits in fp32 to avoid fp16 overflow (tau=0.1 amplifies by 10x)
        logits = torch.bmm(z_anchor.float(), z_key.float().transpose(1, 2)) / self.infonce_tau
        
        # Apply valid mask if provided
        if valid_mask is not None:
            # valid_mask: [B, N]
            # Mask out invalid anchors and keys
            valid_anchor = valid_mask.unsqueeze(2)  # [B, N, 1]
            valid_key = valid_mask.unsqueeze(1)     # [B, 1, N]
            valid_pair = valid_anchor & valid_key   # [B, N, N]
            
            # Mask logits for invalid pairs
            logits = logits.masked_fill(~valid_pair, float('-inf'))
            positive_mask = positive_mask & valid_pair
        
        # Compute InfoNCE loss with multi-positive formulation
        # L_i = -log(sum_j∈P(i) exp(logit_ij)) + log(sum_j exp(logit_ij))
        
        # Mask for numerical stability
        pos_logits = logits.masked_fill(~positive_mask, float('-inf'))
        
        # Log-sum-exp over positives
        log_sum_pos = torch.logsumexp(pos_logits, dim=-1)  # [B, N]
        
        # Log-sum-exp over all (denominator)
        log_sum_all = torch.logsumexp(logits, dim=-1)  # [B, N]
        
        # InfoNCE loss per anchor
        loss_per_anchor = log_sum_all - log_sum_pos  # [B, N]
        
        # Handle cases with no positives (set loss to 0)
        has_positives = positive_mask.any(dim=-1)  # [B, N]
        loss_per_anchor = loss_per_anchor.masked_fill(~has_positives, 0.0)
        
        # Average over valid anchors
        if valid_mask is not None:
            num_valid = (valid_mask & has_positives).sum()
            if num_valid > 0:
                loss = (loss_per_anchor * valid_mask.float() * has_positives.float()).sum() / num_valid
            else:
                loss = torch.tensor(0.0, device=device)
        else:
            num_valid = has_positives.sum()
            if num_valid > 0:
                loss = (loss_per_anchor * has_positives.float()).sum() / num_valid
            else:
                loss = torch.tensor(0.0, device=device)
        
        return loss
    
    def forward(
        self,
        lidar_query: torch.Tensor,
        img_query_noisy: torch.Tensor,
        img_query_corr: torch.Tensor,
        eps_gt: torch.Tensor,
        ref_pos: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for denoising alignment with noise decomposition.
        
        This is called after cross-attention branches have been computed.
        
        Args:
            lidar_query: LiDAR cross-attention output [B, N, C]
            img_query_noisy: Noisy image cross-attention output [B, N, C]
            img_query_corr: Corrected image cross-attention output [B, N, C]
            eps_gt: Ground truth noise [B, noise_dim]
            ref_pos: Reference positions [B, N, 2] or [B, N, 4, 2]
            valid_mask: Optional validity mask [B, N]
            compute_loss: Whether to compute losses
            
        Returns:
            fuse_query: Fused output [B, N, C]
            delta_inj: Injected noise prediction [B, 2]
            delta_sys: Systematic noise prediction [B, 2]
            delta_pred: Total noise prediction [B, 2]
            losses: Dictionary of losses (empty if compute_loss=False)
        """
        # Predict noise decomposition
        delta_inj, delta_sys, delta_pred = self.predict_noise_decomposed(lidar_query, img_query_noisy)
        
        # Fuse queries (residual connection)
        fuse_query = lidar_query + img_query_corr
        
        # Compute losses if requested
        losses = {}
        if compute_loss:
            losses = self.compute_losses(
                delta_inj=delta_inj,
                delta_sys=delta_sys,
                delta_pred=delta_pred,
                eps_gt=eps_gt,
                fuse_query=fuse_query,
                lidar_query=lidar_query,
                img_query_noisy=img_query_noisy,
                ref_pos=ref_pos,
                valid_mask=valid_mask,
            )
        
        return fuse_query, delta_inj, delta_sys, delta_pred, losses


# Unit tests / assertions
def _test_apply_global_noise():
    """Test ApplyGlobalNoise module."""
    print("Testing ApplyGlobalNoise...")
    
    batch_size = 2
    N = 512
    num_corners = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test translation only
    apply_noise = ApplyGlobalNoise(noise_scale=(2.0, 2.0), include_yaw=False)
    
    # Test with [B, N, 4, 2] input
    ref_pos = torch.randn(batch_size, N, num_corners, 2, device=device)
    eps = torch.tensor([[1.0, 0.5], [-0.5, 1.0]], device=device)  # [B, 2]
    
    ref_pos_noisy = apply_noise(ref_pos, eps)
    
    assert ref_pos_noisy.shape == ref_pos.shape, f"Shape mismatch: {ref_pos_noisy.shape}"
    
    # Check that noise was applied correctly
    expected_diff_x = eps[:, 0:1, None, None]  # [B, 1, 1, 1]
    expected_diff_y = eps[:, 1:2, None, None]
    actual_diff_x = (ref_pos_noisy[..., 0] - ref_pos[..., 0]).mean(dim=(1, 2))
    actual_diff_y = (ref_pos_noisy[..., 1] - ref_pos[..., 1]).mean(dim=(1, 2))
    
    assert torch.allclose(actual_diff_x, eps[:, 0], atol=1e-5), "X translation incorrect"
    assert torch.allclose(actual_diff_y, eps[:, 1], atol=1e-5), "Y translation incorrect"
    
    # Test noise sampling
    sampled_noise = apply_noise.sample_noise(batch_size, device)
    assert sampled_noise.shape == (batch_size, 2), f"Sampled noise shape: {sampled_noise.shape}"
    assert (sampled_noise.abs() <= 2.0).all(), "Sampled noise out of range"
    
    print("ApplyGlobalNoise tests passed!")


def _test_denoise_head():
    """Test DenoiseHead module with noise decomposition."""
    print("Testing DenoiseHead...")
    
    batch_size = 2
    N = 512
    C = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    denoise_head = DenoiseHead(
        in_channels=C,
        hidden_channels=256,
        noise_dim=2,
        noise_scale=(2.0, 2.0),
        inj_scale=(2.0, 2.0),
        sys_scale=0.2,
        pool_type='mean',
    ).to(device)
    
    lidar_query = torch.randn(batch_size, N, C, device=device)
    img_query_noisy = torch.randn(batch_size, N, C, device=device)
    
    delta_inj, delta_sys, delta_pred = denoise_head(lidar_query, img_query_noisy)
    
    # Check shapes
    assert delta_inj.shape == (batch_size, 2), f"delta_inj shape: {delta_inj.shape}"
    assert delta_sys.shape == (batch_size, 2), f"delta_sys shape: {delta_sys.shape}"
    assert delta_pred.shape == (batch_size, 2), f"delta_pred shape: {delta_pred.shape}"
    
    # Check delta_inj range (±inj_scale)
    assert (delta_inj.abs() <= 2.0).all(), "delta_inj out of clamping range"
    
    # Check delta_sys magnitude (||delta_sys|| ∈ [0, sys_scale])
    sys_magnitude = delta_sys.norm(dim=-1)
    assert (sys_magnitude <= 0.2 + 1e-5).all(), f"delta_sys magnitude out of range: {sys_magnitude}"
    assert (sys_magnitude >= 0).all(), f"delta_sys magnitude negative: {sys_magnitude}"
    
    # Check delta_pred = delta_inj + delta_sys
    assert torch.allclose(delta_pred, delta_inj + delta_sys, atol=1e-5), "delta_pred != delta_inj + delta_sys"
    
    # Test backward compatible interface
    noise_pred = denoise_head.predict_noise(lidar_query, img_query_noisy)
    assert noise_pred.shape == (batch_size, 2), f"predict_noise shape: {noise_pred.shape}"
    
    print("DenoiseHead tests passed!")


def _test_contrastive_projector():
    """Test ContrastiveProjector module."""
    print("Testing ContrastiveProjector...")
    
    batch_size = 2
    N = 512
    C = 512
    proj_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    projector = ContrastiveProjector(
        in_channels=C,
        proj_channels=proj_dim,
    ).to(device)
    
    x = torch.randn(batch_size, N, C, device=device)
    proj = projector(x)
    
    assert proj.shape == (batch_size, N, proj_dim), f"Projection shape: {proj.shape}"
    
    # Check L2 normalization
    norms = proj.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Not L2 normalized"
    
    print("ContrastiveProjector tests passed!")


def _test_denoising_alignment_module():
    """Test complete DenoisingAlignmentModule with noise decomposition and improvement margin."""
    print("Testing DenoisingAlignmentModule...")
    
    batch_size = 2
    N = 512
    C = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    module = DenoisingAlignmentModule(
        embed_dims=C,
        noise_scale=(2.0, 2.0),
        include_yaw=False,
        sys_scale=0.2,
        proj_channels=128,
        infonce_radius=1.0,
        lambda_inj=1.0,
        lambda_nce=0.1,
        lambda_imp=0.1,
        margin=0.1,
    ).to(device)
    
    lidar_query = torch.randn(batch_size, N, C, device=device)
    img_query_noisy = torch.randn(batch_size, N, C, device=device)
    img_query_corr = torch.randn(batch_size, N, C, device=device)
    eps_gt = torch.randn(batch_size, 2, device=device) * 0.5
    ref_pos = torch.randn(batch_size, N, 2, device=device) * 10  # meters
    
    fuse_query, delta_inj, delta_sys, delta_pred, losses = module(
        lidar_query=lidar_query,
        img_query_noisy=img_query_noisy,
        img_query_corr=img_query_corr,
        eps_gt=eps_gt,
        ref_pos=ref_pos,
        compute_loss=True,
    )
    
    # Check output shapes
    assert fuse_query.shape == (batch_size, N, C), f"Fuse query shape: {fuse_query.shape}"
    assert delta_inj.shape == (batch_size, 2), f"delta_inj shape: {delta_inj.shape}"
    assert delta_sys.shape == (batch_size, 2), f"delta_sys shape: {delta_sys.shape}"
    assert delta_pred.shape == (batch_size, 2), f"delta_pred shape: {delta_pred.shape}"
    
    # Check delta_pred = delta_inj + delta_sys
    assert torch.allclose(delta_pred, delta_inj + delta_sys, atol=1e-5), "delta_pred != delta_inj + delta_sys"
    
    # Check delta_sys magnitude constraint
    sys_magnitude = delta_sys.norm(dim=-1)
    assert (sys_magnitude <= 0.2 + 1e-5).all(), f"delta_sys magnitude out of range: {sys_magnitude}"
    
    # Check required losses
    assert 'loss_inj' in losses, "Missing loss_inj"
    assert 'loss_nce_pre' in losses, "Missing loss_nce_pre"
    assert 'loss_nce_post' in losses, "Missing loss_nce_post"
    assert 'loss_imp' in losses, "Missing loss_imp"
    assert 'loss_alignment' in losses, "Missing loss_alignment"
    assert 'sys_magnitude' in losses, "Missing sys_magnitude monitoring"
    
    # Backward compatibility
    assert 'loss_noise' in losses, "Missing loss_noise (backward compat)"
    assert 'loss_nce' in losses, "Missing loss_nce (backward compat)"
    
    print("DenoisingAlignmentModule tests passed!")


if __name__ == "__main__":
    _test_apply_global_noise()
    _test_denoise_head()
    _test_contrastive_projector()
    _test_denoising_alignment_module()
    print("\nAll tests passed!")

