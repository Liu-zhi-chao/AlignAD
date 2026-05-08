"""
Ref2DSoftTopKUpdater: a soft top-k alignment update module driven by prior anchors.

Before entering the BEVFormerEncoder, this module uses prior anchors to perform a
"soft top-k" alignment update on ref_2d, producing ref_2d_new with exactly the
same shape as the input ref_2d.

Integration example:
In CrossModalRefiner.forward, right after `ref_2d = trajectory_pose.detach()`:
    ref_2d = self.ref2d_updater(bev_query, ref_2d, anchors=self.anchors)
Then pass ref_2d into the encoder; the rest of the logic stays unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Ref2DSoftTopKUpdater(nn.Module):
    """
    Soft top-k alignment update module conditioned on prior anchors.

    The module updates ref_2d in the following steps:
    1. Reshape ref_2d to [B, num_proposals, num_poses, state_dim].
    2. Compute the distance between every proposal and every anchor.
    3. Soft top-k: pick the k nearest anchors and weight them with softmax.
    4. Predict the alpha weight from bev_query.
    5. Fuse ref_2d and anchor_ref: ref_new = (1 - alpha) * ref + alpha * anchor_ref.

    Args:
        embed_dim: feature dim of bev_query, default 512.
        k: k for soft top-k, default 2.
        tau: softmax temperature, default 0.5.
        hidden: hidden dim of the MLP, default 256.
        num_proposals: number of proposals, default 64.
        num_poses: number of poses per proposal, default 8.
        state_dim: per-pose state dimension (x, y, heading), default 3.
        detach_ref: whether to detach ref_2d, default False.
        anchors: optional preset anchors, shape [M, num_poses, state_dim].
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        k: int = 2,
        tau: float = 0.5,
        hidden: int = 256,
        num_proposals: int = 64,
        num_poses: int = 8,
        state_dim: int = 3,
        detach_ref: bool = False,
        anchors: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.k = k
        self.tau = tau
        self.num_proposals = num_proposals
        self.num_poses = num_poses
        self.state_dim = state_dim
        self.detach_ref = detach_ref
        
        # Register anchors as a buffer when provided
        if anchors is not None:
            assert anchors.dim() == 3, f"anchors must be 3D [M, num_poses, state_dim], got {anchors.shape}"
            assert anchors.shape[1] == num_poses, f"anchors.shape[1] must be {num_poses}, got {anchors.shape[1]}"
            assert anchors.shape[2] == state_dim, f"anchors.shape[2] must be {state_dim}, got {anchors.shape[2]}"
            self.register_buffer("anchors", anchors)
        else:
            self.register_buffer("anchors", None)
        
        # Alpha prediction MLP: pooled bev_query -> alpha in (0, 1)
        # Input:  [B, num_proposals, embed_dim]
        # Output: [B, num_proposals, 1]
        self.alpha_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
        )
        
        # Initialize the last bias so the initial alpha is small (conservative update)
        nn.init.zeros_(self.alpha_mlp[-1].weight)
        nn.init.constant_(self.alpha_mlp[-1].bias, -2.0)  # sigmoid(-2) ≈ 0.12
    
    def forward(
        self,
        bev_query: torch.Tensor,
        ref_2d: torch.Tensor,
        anchors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: update ref_2d using prior anchors.

        Args:
            bev_query: BEV query features, shape [B, N, C], where N = num_proposals * num_poses.
            ref_2d:    reference 2D positions, shape [B, N, state_dim], where N = num_proposals * num_poses.
            anchors:   optional anchors, shape [M, num_poses, state_dim]; takes precedence over the buffer.

        Returns:
            ref_2d_new: updated reference 2D positions, shape [B, N, state_dim], with the same shape/dtype/device as the input.
        """
        # ============================================================
        # 1. Validate inputs and prepare
        # ============================================================
        B, N, C = bev_query.shape
        _, N2, D = ref_2d.shape
        
        assert N == self.num_proposals * self.num_poses, \
            f"bev_query.shape[1] must be {self.num_proposals * self.num_poses}, got {N}"
        assert N2 == N, \
            f"ref_2d.shape[1] must match bev_query.shape[1], got {N2} vs {N}"
        assert C == self.embed_dim, \
            f"bev_query.shape[2] must be {self.embed_dim}, got {C}"
        assert D == self.state_dim, \
            f"ref_2d.shape[2] must be {self.state_dim}, got {D}"
        
        # Pick anchors (the argument takes precedence over the buffer)
        if anchors is not None:
            anchors_use = anchors
        elif self.anchors is not None:
            anchors_use = self.anchors
        else:
            raise ValueError("anchors must be provided either in __init__ or forward")
        
        M = anchors_use.shape[0]
        assert anchors_use.shape == (M, self.num_poses, self.state_dim), \
            f"anchors must have shape [M, {self.num_poses}, {self.state_dim}], got {anchors_use.shape}"
        
        # Save the original dtype and device
        orig_dtype = ref_2d.dtype
        device = ref_2d.device
        
        # Make sure anchors live on the same device
        anchors_use = anchors_use.to(device)
        
        # Optional: detach ref_2d
        if self.detach_ref:
            ref_2d = ref_2d.detach()
        
        # ============================================================
        # 2. Reshape ref_2d: [B, N, 3] -> [B, 64, 8, 3]
        # ============================================================
        ref = ref_2d.reshape(B, self.num_proposals, self.num_poses, self.state_dim)
        # ref: [B, 64, 8, 3]
        
        # ============================================================
        # 3. Compute distances: dist[b, p, m] = mean L2 distance
        # ============================================================
        # ref:          [B, 64, 8, 3]
        # anchors_use:  [M, 8, 3]
        # Need the distance from every proposal to every anchor.
        
        # Cast to float32 for numerical stability
        ref_f32 = ref.float()  # [B, 64, 8, 3]
        anchors_f32 = anchors_use.float()  # [M, 8, 3]
        
        # Expand dims for broadcasting:
        # ref_expanded:     [B, 64, 1, 8, 3]
        # anchors_expanded: [1, 1, M, 8, 3]
        ref_expanded = ref_f32.unsqueeze(2)  # [B, 64, 1, 8, 3]
        anchors_expanded = anchors_f32.unsqueeze(0).unsqueeze(0)  # [1, 1, M, 8, 3]
        
        # Squared L2 distance, averaged over pose and state dims
        diff = ref_expanded - anchors_expanded  # [B, 64, M, 8, 3]
        dist_sq = (diff ** 2).mean(dim=(-2, -1))  # [B, 64, M]
        
        # ============================================================
        # 4. Soft top-k selection
        # ============================================================
        # Take the k anchors with the smallest distances.
        # `topk` with largest=False returns the k smallest values.
        k = min(self.k, M)  # guard against k > M
        
        # dist_topk_values:  [B, 64, k]
        # dist_topk_indices: [B, 64, k]
        dist_topk_values, dist_topk_indices = torch.topk(dist_sq, k=k, dim=-1, largest=False)
        
        # Softmax weights with temperature tau.
        # We negate the distance so that smaller distances yield larger weights.
        weights = F.softmax(-dist_topk_values / self.tau, dim=-1)  # [B, 64, k]
        
        # ============================================================
        # 5. Gather the selected anchors and take the weighted sum
        # ============================================================
        # dist_topk_indices: [B, 64, k]
        # We need to gather anchors: [M, 8, 3] -> [B, 64, k, 8, 3]
        
        # Expand indices for gather:
        # indices_expanded: [B, 64, k, 8, 3]
        indices_expanded = dist_topk_indices.unsqueeze(-1).unsqueeze(-1)  # [B, 64, k, 1, 1]
        indices_expanded = indices_expanded.expand(B, self.num_proposals, k, self.num_poses, self.state_dim)
        
        # Expand anchors for gather:
        # anchors_expanded_for_gather: [B, 64, M, 8, 3]
        anchors_expanded_for_gather = anchors_f32.unsqueeze(0).unsqueeze(0).expand(
            B, self.num_proposals, M, self.num_poses, self.state_dim
        )
        
        # Gather the selected anchors
        # anchor_sel: [B, 64, k, 8, 3]
        anchor_sel = torch.gather(anchors_expanded_for_gather, dim=2, index=indices_expanded)
        
        # Weighted sum
        # weights: [B, 64, k] -> [B, 64, k, 1, 1]
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # [B, 64, k, 1, 1]
        
        # anchor_ref: [B, 64, 8, 3]
        anchor_ref = (weights_expanded * anchor_sel).sum(dim=2)  # [B, 64, 8, 3]
        
        # ============================================================
        # 6. Predict the alpha weight
        # ============================================================
        # bev_query: [B, N, C] -> [B, 64, 8, C]
        q = bev_query.reshape(B, self.num_proposals, self.num_poses, self.embed_dim)
        
        # Mean-pool over the pose dim: [B, 64, 8, C] -> [B, 64, C]
        q_pooled = q.mean(dim=2)  # [B, 64, C]
        
        # Predict alpha: [B, 64, C] -> [B, 64, 1]
        alpha_logit = self.alpha_mlp(q_pooled)  # [B, 64, 1]
        alpha = torch.sigmoid(alpha_logit)  # [B, 64, 1]
        
        # Expand alpha for broadcasting: [B, 64, 1] -> [B, 64, 1, 1]
        alpha = alpha.unsqueeze(-1)  # [B, 64, 1, 1]
        
        # ============================================================
        # 7. Fuse and update
        # ============================================================
        # ref:        [B, 64, 8, 3]
        # anchor_ref: [B, 64, 8, 3]
        # alpha:      [B, 64, 1, 1] broadcasts to [B, 64, 8, 3]
        
        ref_new = (1.0 - alpha) * ref_f32 + alpha * anchor_ref  # [B, 64, 8, 3]
        
        # ============================================================
        # 8. Reshape back to the original layout and restore dtype
        # ============================================================
        ref_2d_new = ref_new.reshape(B, N, self.state_dim)  # [B, N, 3]
        
        # Restore the original dtype
        ref_2d_new = ref_2d_new.to(orig_dtype)
        
        return ref_2d_new
    
    def get_alpha(
        self,
        bev_query: torch.Tensor,
    ) -> torch.Tensor:
        """
        Helper: return only the alpha weights, useful for visualization or debugging.

        Args:
            bev_query: BEV query features, shape [B, N, C].

        Returns:
            alpha: weights, shape [B, num_proposals].
        """
        B = bev_query.shape[0]
        
        # bev_query: [B, N, C] -> [B, 64, 8, C]
        q = bev_query.reshape(B, self.num_proposals, self.num_poses, self.embed_dim)
        
        # Mean-pool over the pose dim: [B, 64, 8, C] -> [B, 64, C]
        q_pooled = q.mean(dim=2)  # [B, 64, C]
        
        # Predict alpha: [B, 64, C] -> [B, 64, 1]
        alpha_logit = self.alpha_mlp(q_pooled)  # [B, 64, 1]
        alpha = torch.sigmoid(alpha_logit).squeeze(-1)  # [B, 64]
        
        return alpha
    
    def get_anchor_distances(
        self,
        ref_2d: torch.Tensor,
        anchors: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper: return the per-proposal distance to every anchor, useful for visualization or debugging.

        Args:
            ref_2d: reference 2D positions, shape [B, N, state_dim].
            anchors: optional anchors, shape [M, num_poses, state_dim].

        Returns:
            dist: distance matrix, shape [B, num_proposals, M].
            topk_indices: top-k anchor indices, shape [B, num_proposals, k].
        """
        B, N, D = ref_2d.shape
        
        # Pick anchors
        if anchors is not None:
            anchors_use = anchors
        elif self.anchors is not None:
            anchors_use = self.anchors
        else:
            raise ValueError("anchors must be provided")
        
        M = anchors_use.shape[0]
        device = ref_2d.device
        anchors_use = anchors_use.to(device)
        
        # Reshape ref_2d
        ref = ref_2d.reshape(B, self.num_proposals, self.num_poses, self.state_dim)
        
        # Compute distances
        ref_f32 = ref.float()
        anchors_f32 = anchors_use.float()
        
        ref_expanded = ref_f32.unsqueeze(2)  # [B, 64, 1, 8, 3]
        anchors_expanded = anchors_f32.unsqueeze(0).unsqueeze(0)  # [1, 1, M, 8, 3]
        
        diff = ref_expanded - anchors_expanded  # [B, 64, M, 8, 3]
        dist = (diff ** 2).mean(dim=(-2, -1))  # [B, 64, M]
        
        # Top-k indices
        k = min(self.k, M)
        _, topk_indices = torch.topk(dist, k=k, dim=-1, largest=False)
        
        return dist, topk_indices


# ============================================================
# Unit test
# ============================================================
def _test_ref2d_soft_topk_updater():
    """A small unit test."""
    import numpy as np
    
    print("Testing Ref2DSoftTopKUpdater...")
    
    # Parameters
    B = 2
    num_proposals = 64
    num_poses = 8
    state_dim = 3
    embed_dim = 512
    M = 20  # number of anchors
    k = 2
    
    N = num_proposals * num_poses  # 512
    
    # Build test data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    bev_query = torch.randn(B, N, embed_dim, device=device, dtype=dtype)
    ref_2d = torch.randn(B, N, state_dim, device=device, dtype=dtype)
    anchors = torch.randn(M, num_poses, state_dim, device=device, dtype=dtype)
    
    # Build the module
    updater = Ref2DSoftTopKUpdater(
        embed_dim=embed_dim,
        k=k,
        tau=0.5,
        hidden=256,
        num_proposals=num_proposals,
        num_poses=num_poses,
        state_dim=state_dim,
        detach_ref=False,
        anchors=anchors,
    ).to(device)
    
    # Forward pass
    ref_2d_new = updater(bev_query, ref_2d)
    
    # Validate the output
    assert ref_2d_new.shape == ref_2d.shape, f"Shape mismatch: {ref_2d_new.shape} vs {ref_2d.shape}"
    assert ref_2d_new.dtype == ref_2d.dtype, f"Dtype mismatch: {ref_2d_new.dtype} vs {ref_2d.dtype}"
    assert ref_2d_new.device == ref_2d.device, f"Device mismatch: {ref_2d_new.device} vs {ref_2d.device}"
    
    # Test gradient flow
    loss = ref_2d_new.sum()
    loss.backward()
    
    # Inspect gradients
    for name, param in updater.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            print(f"  {name}: grad norm = {param.grad.norm().item():.6f}")
    
    print(f"  Input shape: {ref_2d.shape}")
    print(f"  Output shape: {ref_2d_new.shape}")
    print(f"  Anchors shape: {anchors.shape}")
    
    # Test the helper methods
    alpha = updater.get_alpha(bev_query)
    assert alpha.shape == (B, num_proposals), f"Alpha shape mismatch: {alpha.shape}"
    print(f"  Alpha shape: {alpha.shape}")
    print(f"  Alpha range: [{alpha.min().item():.4f}, {alpha.max().item():.4f}]")
    
    dist, topk_idx = updater.get_anchor_distances(ref_2d)
    assert dist.shape == (B, num_proposals, M), f"Dist shape mismatch: {dist.shape}"
    assert topk_idx.shape == (B, num_proposals, k), f"Topk idx shape mismatch: {topk_idx.shape}"
    print(f"  Distance matrix shape: {dist.shape}")
    print(f"  Top-k indices shape: {topk_idx.shape}")
    
    # Make sure the anchors argument takes precedence
    anchors_override = torch.randn(M, num_poses, state_dim, device=device, dtype=dtype)
    ref_2d_new_override = updater(bev_query, ref_2d, anchors=anchors_override)
    assert ref_2d_new_override.shape == ref_2d.shape
    
    print("All tests passed!")


if __name__ == "__main__":
    _test_ref2d_soft_topk_updater()
