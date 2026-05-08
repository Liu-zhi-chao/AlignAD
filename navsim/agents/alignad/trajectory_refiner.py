import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import torch
from mmengine.registry import MODELS
from .encoder.cross_modal_refiner import CrossModalRefiner
from .encoder.transformer_decoder import MLP
from .alignad_config import AlignADConfig

class TrajectoryRefiner(nn.Module):
    def __init__(self, config: AlignADConfig, enable_denoising: bool = False):
        super().__init__()

        self.poses_num = config.num_poses
        self.state_size = 3
        self.enable_denoising = enable_denoising

        self.traj_decoder = MLP(config.tf_d_model, config.tf_d_ffn, 3)

        self.refiner = CrossModalRefiner(config, enable_denoising=enable_denoising)

    def forward(
        self, 
        bev_query: torch.Tensor, 
        proposal_list: Optional[List[torch.Tensor]], 
        image_feature, 
        pts_feature,
        training: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass for trajectory refinement.
        
        Args:
            bev_query: BEV query features [B, N, C]
            proposal_list: List of previous proposals
            image_feature: Image features tuple
            pts_feature: Point cloud features tuple
            training: Whether in training mode
            
        Returns:
            bev_query: Updated BEV query features
            proposal_list: Updated list of proposals
            denoising_output: Dictionary of denoising outputs (empty if denoising disabled)
        """
        if proposal_list is None:
            proposal_list = []
        
        proposals = self.traj_decoder(bev_query).reshape(
            bev_query.shape[0], -1, self.poses_num, self.state_size
        )  # bs, config.proposal_num, 8, 3

        proposal_list.append(proposals)

        bev_query, denoising_output = self.refiner(
            image_feature, pts_feature, proposals, bev_query, training=training
        )

        return bev_query, proposal_list, denoising_output
        