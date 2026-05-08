from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from .score_module.scorer import Scorer
from .trajectory_refiner import TrajectoryRefiner
from .encoder.sensor_backbone import SensorBackbone
from .alignad_config import AlignADConfig

class AlignADModel(nn.Module):
    def __init__(self, config: AlignADConfig, enable_denoising: bool = False):
        super().__init__()
        self._config = config
        self.poses_num = config.num_poses
        self.state_size = 3
        self.enable_denoising = enable_denoising
        self._backbone = SensorBackbone(config)

        self._ego_encoding = nn.Linear(11, config.tf_d_model)

        self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)   # 8 * 64 = 512, 512

        self.refiner_num = config.refiner_num

        self._trajectory_refiners = nn.ModuleList([
            TrajectoryRefiner(config, enable_denoising=enable_denoising) 
            for _ in range(self.refiner_num)
        ])


        self._scorer = Scorer(config)

        self.b2d=config.b2d

    def forward(
        self, 
        features: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor],
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the AlignADModel.
        
        Args:
            features: Dictionary of input features.
            targets: Dictionary of target data.
            training: Whether in training mode (affects noise injection for denoising).
        
        Returns:
            Dictionary of output predictions and scores.
        """
        batch_size, multi_feature, bev_query = self._process_inputs(features)

        proposal_list = []
        all_denoising_outputs: List[Dict[str, torch.Tensor]] = []
        
        for i, refiner in enumerate(self._trajectory_refiners):
            bev_query, proposal_list, denoising_output = refiner(
                bev_query, proposal_list, multi_feature[2:6], multi_feature[6:], training=training
            )
            if denoising_output:
                all_denoising_outputs.append(denoising_output)

        proposals = proposal_list[-1]    # bs, 64, 8, 3

        output = {}
        output["proposal_list"] = proposal_list

        output.update(self._score_proposals(proposals, bev_query))
        
        output.update(self._compute_trajectory(output["proposals"], output["pred_logit"], batch_size))
        
        # Aggregate denoising outputs and losses
        if self.enable_denoising and all_denoising_outputs:
            output.update(self._aggregate_denoising_outputs(all_denoising_outputs))
        
        return output
    
    def _aggregate_denoising_outputs(
        self, 
        all_denoising_outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate denoising outputs from all refiner layers.
        
        Handles noise decomposition (inj + sys) and improvement margin losses:
        - loss_inj: Injection noise regression (delta_inj vs eps_gt)
        - loss_nce_pre: InfoNCE before correction
        - loss_nce_post: InfoNCE after correction  
        - loss_imp: Improvement margin loss
        - loss_alignment: Total alignment loss
        - sys_magnitude: Systematic noise magnitude (for monitoring)
        
        Args:
            all_denoising_outputs: List of denoising output dicts from each refiner
            
        Returns:
            Aggregated denoising outputs and losses
        """
        aggregated = {}
        
        # Collect all loss keys and monitoring keys
        loss_keys = set()
        monitor_keys = set()
        for output in all_denoising_outputs:
            for key in output.keys():
                if key.startswith('loss_'):
                    loss_keys.add(key)
                elif key == 'sys_magnitude':  # Special monitoring key
                    monitor_keys.add(key)
        
        # Average losses across refiners
        for key in loss_keys:
            values = []
            for output in all_denoising_outputs:
                if key in output:
                    val = output[key]
                    if isinstance(val, torch.Tensor):
                        values.append(val)
            if values:
                aggregated[f'denoising_{key}'] = torch.stack(values).mean()
        
        # Average monitoring metrics across refiners
        for key in monitor_keys:
            values = []
            for output in all_denoising_outputs:
                if key in output:
                    val = output[key]
                    if isinstance(val, torch.Tensor):
                        values.append(val)
            if values:
                aggregated[f'denoising_{key}'] = torch.stack(values).mean()
        
        # Keep last refiner's non-loss outputs (delta_inj, delta_sys, etc.)
        last_output = all_denoising_outputs[-1]
        for key, value in last_output.items():
            if not key.startswith('loss_') and key not in monitor_keys:
                aggregated[f'denoising_{key}'] = value
        
        # Compute total denoising loss (only from loss_alignment, not summing individual components)
        # loss_alignment already contains: loss_inj + loss_nce_post + loss_imp
        if 'denoising_loss_alignment' in aggregated:
            aggregated['denoising_loss_total'] = aggregated['denoising_loss_alignment']
        else:
            # Fallback: sum all loss components
            total_loss = torch.tensor(0.0, device=next(iter(last_output.values())).device)
            for key, value in aggregated.items():
                if key.startswith('denoising_loss_') and isinstance(value, torch.Tensor):
                    if key not in ['denoising_loss_total', 'denoising_loss_nce_pre']:  # Don't double count
                        total_loss = total_loss + value
            aggregated['denoising_loss_total'] = total_loss
        
        return aggregated
    
    def _process_inputs(self, features):
        """Process input features."""
        ego_status = features["ego_status"][:, -1]  # bs, 11
        if self.b2d:
                ego_status[:, 1:3] = 0
        camera_feature = features["camera_feature"]  # bs, n_cam, 3, 224, 384
        
        # Normalize lidar_feature dtype/container (it may be a list, numpy array, or torch.Tensor)
        lidar_feature = features["lidar_feature"]
        if isinstance(lidar_feature, list):
            # If the list holds tensors, move them to CPU before stacking/conversion
            if len(lidar_feature) > 0 and isinstance(lidar_feature[0], torch.Tensor):
                # List contains tensors; move to CPU first
                if len(lidar_feature) == 1:
                    lidar_feature = lidar_feature[0].cpu()
                else:
                    lidar_feature = torch.stack([t.cpu() if isinstance(t, torch.Tensor) else torch.tensor(t) for t in lidar_feature])
            else:
                # List of plain numeric values; try converting via numpy first.
                # Fall back to a manual CPU copy if it contains CUDA tensors.
                try:
                    lidar_feature = torch.tensor(np.array(lidar_feature, dtype=np.float32))
                except (TypeError, RuntimeError):
                    # Conversion failed (likely CUDA tensors inside); move them to CPU first
                    cpu_list = []
                    for t in lidar_feature:
                        if isinstance(t, torch.Tensor):
                            cpu_list.append(t.cpu().numpy())
                        else:
                            cpu_list.append(t)
                    lidar_feature = torch.from_numpy(np.array(cpu_list, dtype=np.float32))
        elif isinstance(lidar_feature, np.ndarray):
            lidar_feature = torch.from_numpy(lidar_feature.astype(np.float32))
        elif not isinstance(lidar_feature, torch.Tensor):
            # Not a list / numpy array / torch.Tensor: try a generic conversion
            lidar_feature = torch.tensor(lidar_feature)
        
        # Make sure lidar_feature is on the right device (matching camera_feature or ego_status)
        if isinstance(camera_feature, torch.Tensor):
            lidar_feature = lidar_feature.to(camera_feature.device)
        elif isinstance(ego_status, torch.Tensor):
            lidar_feature = lidar_feature.to(ego_status.device)
        
        lidar_feature = lidar_feature.transpose(-2, -1).contiguous()  # bs, 1, 256, 256
        batch_size = ego_status.shape[0]
        
        ego_feature = self._ego_encoding(ego_status)    # bs, tf_d_model

        trajectory_query = ego_feature[:,None] + self.init_feature.weight[None]  # bs, 512, 512

        multi_feature = self._backbone(camera_feature, lidar_feature, img_metas=features)
        
        return batch_size, multi_feature, trajectory_query


    def _bev_semantic_map(self, bev_query, targets):
        """Predict BEV semantic map."""
        # bev_map_pred = self.map_head(bev_query)  # bs, num_classes, H, W

        bev_map_pred = None
        
        width = targets["bev_occ_label"].shape[1]
        bev_occ_label = targets["bev_occ_label"][:, width//2:, ...]
        # bev_occ_label = targets["bev_occ_label"][:, width//2:, ...].flatten(1).contiguous()

        return {
            "bev_map_pred": bev_map_pred,
            "bev_map_gt": bev_occ_label if bev_map_pred is not None else None,
        }


    def _decode_traj(self, proposal_list, targets, bev_query):
        """Decode proposal features into occupancy maps."""
        occ_xy = targets["bev_occ_xy"]

        proposal_dict = self._trajectory_decoder(
            proposal_list,
            occ_xy=occ_xy if targets is not None else None,
            occ_label=targets["bev_occ_label"] if targets is not None else None,
            bev_query=bev_query,  # Pass trajectory features for semantic prediction
        )
        return {
            "bev_map_pred": proposal_dict["pred_occ"],
            "bev_map_gt": proposal_dict["sampled_label"],
        }
    
    
    def _score_proposals(self, proposals, bev_query):
        """Score proposals and compute agent states."""

        pred_logit, pred_agents_states, pred_area_logit  = self._scorer(proposals, bev_query)  # use proposal_feature to get each trajectory's score
        return {
            "proposals": proposals,   # b, 64, 8, 3
            # "proposal_list": proposal_list,
            "pred_logit": pred_logit,  # b, 64, 6
            "pred_agents_states": pred_agents_states,
            "pred_area_logit": pred_area_logit
        }
    
    def _compute_trajectory(self, proposals, pred_logit, batch_size):
        """Compute the final trajectory and PDM score."""
        pdm_score = torch.sigmoid(pred_logit)[:, :, -1] # b, 64
        
        token = torch.argmax(pdm_score, dim=1)  # b,
        trajectory = proposals[torch.arange(batch_size), token] # b, num_pose, 3
        return {
            "trajectory": trajectory,
            "pdm_score": pdm_score,
        }