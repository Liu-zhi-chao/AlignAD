from typing import Any, List, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from pathlib import Path
import pickle
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.dataset import load_feature_target_from_pickle
from pytorch_lightning.callbacks import ModelCheckpoint
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.agents.alignad.alignad_model import AlignADModel
from navsim.agents.alignad.alignad_features import AlignADTargetBuilder
from navsim.agents.alignad.alignad_features import AlignADFeatureBuilder
from navsim.agents.alignad.alignad_config import AlignADConfig
from navsim.agents.alignad.encoder.utils import dict_to_gpu
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from navsim.agents.alignad.alignad_loss import _bev_occ_loss
from navsim.agents.alignad.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim
from .score_module.compute_b2d_score import compute_corners_torch

def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop("type")
    return getattr(obj, type)(**cfg, **kwargs)

class AlignADAgent(AbstractAgent):    
    def __init__(
            self,
            config: AlignADConfig,
            lr: float,
            checkpoint_path: str = None,
    ):
        """
        Initialize AlignAD Agent
        
        Args:
            config: Configuration object containing model hyperparameters
            lr: Learning rate for optimization
            checkpoint_path: Path to pre-trained model checkpoint
        """
        super().__init__()
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path

        cache_data = config.cache_data  # Whether to use cached data only
        
        # Get denoising flag from config
        enable_denoising = getattr(config, 'enable_denoising', False)

        if not cache_data:
            # Initialize the main AlignAD model
            self._alignad_model = AlignADModel(config, enable_denoising=enable_denoising)
            self.initialize()

        if not cache_data:  # Only for training
            # Initialize loss functions
            self.bce_logit_loss = nn.BCEWithLogitsLoss()  # Binary cross-entropy for scoring
            # Enable Ray for parallel processing
            self.ray = True

            self.b2d = config.b2d

            if self.ray:
                from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
                from nuplan.planning.utils.multithreading.worker_utils import worker_map
                if self.b2d:
                    self.worker = RayDistributedNoTorch(threads_per_node=24, log_to_driver=False)
                else:
                    self.worker = RayDistributedNoTorch(threads_per_node=48, log_to_driver=False)
                self.worker_map = worker_map

            if config.b2d:
                import gzip
                with gzip.open(os.getenv("NAVSIM_EXP_ROOT") + "/B2d_cache/train_fut_boxes.gz", "rb") as f:
                    self.train_metric_cache_paths = pickle.load(f)
                with gzip.open(os.getenv("NAVSIM_EXP_ROOT") + "/B2d_cache/val_fut_boxes.gz", "rb") as f:
                    self.test_metric_cache_paths = pickle.load(f)
                from .score_module.compute_b2d_score import get_scores
                self.get_scores = get_scores

                map_file =os.getenv("NAVSIM_EXP_ROOT") +"/map.pkl"

                with open(map_file, 'rb') as f:
                    self.map_infos = pickle.load(f)
                self.cuda_map=False
            else:
                # Load NavSim metric cache
                from .score_module.compute_navsim_score import get_scores
                # print(os.getenv("NAVSIM_EXP_ROOT"))
                metric_cache = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_cache"))
                self.train_metric_cache_paths = metric_cache.metric_cache_paths
                self.test_metric_cache_paths = metric_cache.metric_cache_paths
                self.get_scores = get_scores

    def name(self) -> str:
        """Return the agent's class name for identification."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """
        Initialize the agent by loading pre-trained weights if available.
        This is called after the agent is instantiated but before inference.
        """
        if self._checkpoint_path != "":
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                    "state_dict"]

            state_dict = {
                k.replace("agent._aligndrive_model", "_alignad_model")
                 .replace("agent._alignad_model", "_alignad_model"): v
                for k, v in state_dict.items()
            }
            
            # Compare loaded state_dict with model's current state_dict
            model_state_dict = self.state_dict()
            loaded_keys = set(state_dict.keys())
            model_keys = set(model_state_dict.keys())
            
            # Print differences
            print("=" * 60)
            print("STATE_DICT COMPARISON")
            print("=" * 60)
            
            # Keys only in loaded state_dict
            only_in_loaded = loaded_keys - model_keys
            if only_in_loaded:
                print(f"\nKeys only in LOADED state_dict ({len(only_in_loaded)} keys):")
                for key in sorted(only_in_loaded):
                    print(f"  - {key}")
            
            # Keys only in model state_dict
            only_in_model = model_keys - loaded_keys
            if only_in_model:
                print(f"\nKeys only in MODEL state_dict ({len(only_in_model)} keys):")
                for key in sorted(only_in_model):
                    print(f"  - {key}")
            
            # Keys with different shapes
            common_keys = loaded_keys & model_keys
            shape_mismatches = []
            for key in common_keys:
                if state_dict[key].shape != model_state_dict[key].shape:
                    shape_mismatches.append((key, state_dict[key].shape, model_state_dict[key].shape))
            
            if shape_mismatches:
                print(f"\nKeys with SHAPE mismatches ({len(shape_mismatches)} keys):")
                for key, loaded_shape, model_shape in shape_mismatches:
                    print(f"  - {key}: loaded{loaded_shape} vs model{model_shape}")
            
            print(f"\nSummary:")
            print(f"  Total loaded keys: {len(loaded_keys)}")
            print(f"  Total model keys: {len(model_keys)}")
            print(f"  Common keys: {len(common_keys)}")
            print(f"  Keys only in loaded: {len(only_in_loaded)}")
            print(f"  Keys only in model: {len(only_in_model)}")
            print(f"  Shape mismatches: {len(shape_mismatches)}")
            print("=" * 60)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    "Checkpoint key mismatch when loading "
                    f"{self._checkpoint_path}. "
                    f"missing={len(missing)} (e.g. {list(missing)[:3]}), "
                    f"unexpected={len(unexpected)} (e.g. {list(unexpected)[:3]}). "
                    "Make sure the checkpoint was trained with a matching "
                    "model definition, or update the prefix mapping above."
                )

    def model_to_cuda(self):
        self._alignad_model.cuda()

    def get_sensor_config(self):
        """
        Define the sensor configuration required by the agent.
        
        Returns:
            SensorConfig: Configuration specifying which cameras and sensors to use
                - cam_f0: Front camera (index 3)
                - cam_l0, cam_r0, cam_b0: Left, right, back cameras (index 3)
                - Other cameras and lidar disabled
        """
        return SensorConfig(
            cam_f0=[3],     # Front camera at index 3
            cam_l0=[3],     # Left camera at index 3
            cam_l1=[],      # Left camera 1 disabled
            cam_l2=[],      # Left camera 2 disabled
            cam_r0=[3],     # Right camera at index 3
            cam_r1=[],      # Right camera 1 disabled
            cam_r2=[],      # Right camera 2 disabled
            cam_b0=[],     # Back camera at index 3
            lidar_pc=[3],    # LiDAR point cloud disabled
        )
    def get_target_builders(self):
        """
        Return target builders for creating training targets.
        
        Returns:
            List of target builders used to process ground truth data
        """
        return [AlignADTargetBuilder(config=self._config)]

    def get_feature_builders(self):
        """
        Return feature builders for processing input features.
        
        Returns:
            List of feature builders used to process sensor inputs
        """
        return [AlignADFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the AlignAD model.

        Args:
            features: Dictionary of input features
                - Camera images: shape [B, N_cam, C, H, W]
                - Other sensor data as configured
        
        Returns:
            Dictionary containing:
                - proposals: Generated trajectory proposals [B, N_proposals, T, 3]
                - pdm_score: Proposal scoring logits [B, N_proposals]
                - Other model outputs
        """
        return self._alignad_model(features, targets, training=self.training)

    def compute_trajectory(self, scene):
       
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(scene.get_agent_input()))

        for builder in self.get_target_builders():
            targets.update(builder.compute_targets(scene))

        targets.pop("token")
        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}
        targets = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

        features = dict_to_gpu(features)
        targets = dict_to_gpu(targets)
        # forward pass
        with torch.no_grad():
            predictions = self.forward(features, targets)
            poses = predictions["trajectory"].squeeze(0).cpu().numpy()

        # extract trajectory
        return Trajectory(poses)
    
    def compute_score(self, targets, proposals, test=True):
        """
        Compute safety and feasibility scores for trajectory proposals.
        
        Args:
            targets: Dictionary containing:
                - trajectory: Ground truth trajectory [B, T, 3]
                - token: Scene tokens [B]
                - town_name: Map names [B]
                - lidar2world: Transform matrices [B, 4, 4]
            proposals: Predicted trajectory proposals [B, N_proposals, T, 3]
            test: Whether in test mode (affects metrics computed)
        
        Returns:
            If test=True:
                - final_scores: Final safety scores [B, N_proposals]
                - best_scores: Best score per sample [B]
                - all_scores: All intermediate scores [B, N_proposals, T]
                - l2_2s: L2 distance for first 2 seconds [B]
                - target_scores: Ground truth scores [B]
            If test=False:
                - final_scores: Final safety scores [B, N_proposals]
                - best_scores: Best score per sample [B]
                - target_scores: All scores including intermediate [B, N_proposals, T]
                - key_agent_corners: Agent corner coordinates [B, N_agents, 4, 2]
                - key_agent_labels: Agent validity labels [B, N_agents]
                - all_ego_areas: Area compliance flags [B, N_proposals, T, 3]
        """
        # Select appropriate metric cache based on training/testing mode
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
        else:
            metric_cache_paths = self.test_metric_cache_paths

        target_trajectory = targets["trajectory"]  # [B, T, 3]
        proposals = proposals.detach()  # [B, N_proposals, T, 3]

        # print("b2d value: ", self.b2d)

        if self.b2d:
            data_points = []

            lidar2worlds=targets["lidar2world"]

            all_proposals = torch.cat([proposals, target_trajectory[:,None]], dim=1)

            all_proposals_xy=all_proposals[:, :,:, :2]
            all_proposals_heading=all_proposals[:, :,:, 2:]

            all_pos = all_proposals_xy.reshape(len(target_trajectory),-1, 2)

            mid_points = (all_pos.amax(1) + all_pos.amin(1)) / 2

            dists = torch.linalg.norm(all_pos - mid_points[:,None], dim=-1).amax(1) + 5

            xyz = torch.cat(
                [mid_points[..., :2], torch.zeros_like(mid_points[..., :1]), torch.ones_like(mid_points[..., :1])], dim=-1)

            xys = torch.einsum("nij,nj->ni", lidar2worlds, xyz)[:, :2]

            vel=torch.cat([all_proposals_xy[:, :,:1], all_proposals_xy[:,:, 1:] - all_proposals_xy[:,:, :-1]],dim=2)/ 0.5

            proposals_05 = torch.cat([all_proposals_xy + vel*0.5, all_proposals_heading], dim=-1)

            proposals_1 = torch.cat([all_proposals_xy + vel*1, all_proposals_heading], dim=-1)

            proposals_ttc = torch.stack([all_proposals, proposals_05,proposals_1], dim=3)

            ego_corners_ttc = compute_corners_torch(proposals_ttc.reshape(-1, 3)).reshape(proposals_ttc.shape[0],proposals_ttc.shape[1], proposals_ttc.shape[2], proposals_ttc.shape[3],  4, 2)

            ego_corners_center = torch.cat([ego_corners_ttc[:,:,:,0], all_proposals_xy[:, :, :, None]], dim=-2)

            ego_corners_center_xyz = torch.cat(
                [ego_corners_center, torch.zeros_like(ego_corners_center[..., :1]), torch.ones_like(ego_corners_center[..., :1])], dim=-1)

            global_ego_corners_centers = torch.einsum("nij,nptkj->nptki", lidar2worlds, ego_corners_center_xyz)[..., :2]

            accs = torch.linalg.norm(vel[:,:, 1:] - vel[:,:, :-1], dim=-1) / 0.5

            turning_rate=torch.abs(torch.cat([all_proposals_heading[:, :,:1,0]-np.pi/2, all_proposals_heading[:,:, 1:,0]-all_proposals_heading[:,:, :-1,0]],dim=2)) / 0.5

            comforts = (accs[:,:-1] < accs[:,-1:].max()).all(-1) & (turning_rate[:,:-1] < turning_rate[:,-1:].max()).all(-1)

            if self.cuda_map==False:
                for key, value in self.map_infos.items():
                    self.map_infos[key] = torch.tensor(value).to(target_trajectory.device)
                self.cuda_map=True

            for token, town_name, proposal,target_traj, comfort, dist, xy,global_conners,local_corners in zip(targets["token"], targets["town_name"], proposals.cpu().numpy(),  target_trajectory.cpu().numpy(), comforts.cpu().numpy(), dists.cpu().numpy(), xys, global_ego_corners_centers,ego_corners_ttc.cpu().numpy()):
                all_lane_points = self.map_infos[town_name[:6]]

                dist_to_cur = torch.linalg.norm(all_lane_points[:,:2] - xy, dim=-1)

                nearby_point = all_lane_points[dist_to_cur < dist]

                lane_xy = nearby_point[:, :2]
                lane_width = nearby_point[:, 2]
                lane_id = nearby_point[:, -1]

                dist_to_lane = torch.linalg.norm(global_conners[None] - lane_xy[:, None, None, None], dim=-1)

                on_road = dist_to_lane < lane_width[:, None, None, None]

                on_road_all = on_road.any(0).all(-1)

                nearest_lane = torch.argmin(dist_to_lane - lane_width[:, None, None,None], dim=0)

                nearest_lane_id=lane_id[nearest_lane]

                center_nearest_lane_id=nearest_lane_id[:,:,-1]

                nearest_road_id = torch.round(center_nearest_lane_id)

                target_road_id = torch.unique(nearest_road_id[-1])

                on_route_all = torch.isin(nearest_road_id, target_road_id)
                # in_multiple_lanes: if
                # - more than one drivable polygon contains at least one corner
                # - no polygon contains all corners
                corner_nearest_lane_id=nearest_lane_id[:,:,:-1]

                batch_multiple_lanes_mask = (corner_nearest_lane_id!=corner_nearest_lane_id[:,:,:1]).any(-1)

                on_road_all=on_road_all==on_road_all[-1:]
                # on_road_all = on_road_all | ~on_road_all[-1:]# on road or groundtruth offroad

                ego_areas=torch.stack([batch_multiple_lanes_mask,on_road_all,on_route_all],dim=-1)

                data_dict = {
                    "fut_box_corners": metric_cache_paths[token],
                    "_ego_coords": local_corners,
                    "target_traj": target_traj,
                    "proposal":proposal,
                    "comfort": comfort,
                    "ego_areas": ego_areas.cpu().numpy(),
                }
                data_points.append(data_dict)
        else:
            # NavSim scoring: simpler trajectory evaluation
            data_points = [
                {
                    "token": metric_cache_paths[token],  # Metric cache token
                    "poses": poses,                      # Trajectory poses [N_proposals, T, 3]
                    "test": test                         # Test mode flag
                }
                for token, poses in zip(targets["token"], proposals.cpu().numpy())
            ]


        # Compute scores using parallel processing or sequential
        if self.ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        # Extract target scores from results
        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)  # [B, N_proposals, 6]
        final_scores = target_scores[:, :, -1]  # [B, N_proposals] - final time step scores
        best_scores = torch.amax(final_scores, dim=-1)  # [B] - best score per sample

        if test:
            return final_scores.mean(), best_scores.mean(), final_scores
        else:
            key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)

            key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)

            all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)

            return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas

    def score_loss(self, pred_logit, agents_state, pred_area_logits, target_scores, gt_states, gt_valid, gt_ego_areas):
        """
        Compute scoring loss for trajectory proposals and agent predictions.
        
        Args:
            pred_logit: Primary score predictions [B, N_proposals, T]
            pred_logit2: Secondary score predictions [B, N_proposals, T] (optional)
            agents_state: Agent state predictions [B, N_agents, state_dim] (optional)
            pred_area_logits: Area compliance predictions [B, N_proposals, T, 3] (optional)
            target_scores: Ground truth scores [B, N_proposals, T]
            gt_states: Ground truth agent states [B, N_agents, state_dim]
            gt_valid: Agent validity mask [B, N_agents]
            gt_ego_areas: Ground truth area compliance [B, N_proposals, T, 3]
        
        Returns:
            Tuple of losses:
                - sub_score_loss: Loss on intermediate scores
                - final_score_loss: Loss on final time step scores
                - pred_ce_loss: Agent existence classification loss
                - pred_l1_loss: Agent state regression loss
                - pred_area_loss: Area compliance classification loss
        """

        if agents_state is not None:
            pred_states = agents_state[..., :-1].reshape(gt_states.shape)
            pred_logits = agents_state[..., -1:].reshape(gt_valid.shape)

            pred_l1_loss = F.l1_loss(pred_states, gt_states, reduction="none")[gt_valid]

            if len(pred_l1_loss):
                pred_l1_loss = pred_l1_loss.mean()
            else:
                pred_l1_loss = pred_states.mean() * 0

            pred_ce_loss = F.binary_cross_entropy_with_logits(pred_logits, gt_valid.to(torch.float32), reduction="mean")

        else:
            pred_ce_loss = 0
            pred_l1_loss = 0

        if pred_area_logits is not None:
            pred_area_logits = pred_area_logits.reshape(gt_ego_areas.shape)

            pred_area_loss = F.binary_cross_entropy_with_logits(pred_area_logits, gt_ego_areas.to(torch.float32),
                                                              reduction="mean")
        else:
            pred_area_loss = 0

        # Primary scoring loss on intermediate time steps
        sub_score_loss = self.bce_logit_loss(pred_logit, target_scores[..., -pred_logit.shape[-1]:])
        
        # Primary scoring loss on final time step
        final_score_loss = self.bce_logit_loss(pred_logit[..., -1], target_scores[..., -1])

        return sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss

    def diversity_loss(self, proposals):
        """
        Compute diversity loss to encourage diverse trajectory proposals.
        
        Args:
            proposals: Trajectory proposals [B, N_proposals, T, 3]
        
        Returns:
            inter_loss: Negative minimum inter-proposal distance (encourages diversity)
                       Shape: scalar
        """
        # Compute pairwise L1 distances between all proposals
        dist = torch.linalg.norm(proposals[:, :, None] - proposals[:, None], dim=-1, ord=1).mean(-1)  # [B, N_proposals, N_proposals]
        
        # Add small value to avoid division by zero
        dist = dist + (dist == 0)  # [B, N_proposals, N_proposals]
        
        # Find minimum distance for each proposal (excluding self-distance)
        inter_loss = -dist.amin(1).amin(1).mean()  # Negative to encourage larger distances
        
        return inter_loss

    def alignad_loss(self, targets: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], config):
        """
        Compute the complete AlignAD loss including trajectory, scoring, and auxiliary losses.
        
        Args:
            targets: Dictionary of ground truth data:
                - trajectory: Ground truth trajectory [B, T, 3]
                - Other target data for scoring and auxiliary tasks
            pred: Dictionary of model predictions:
                - proposals: Final trajectory proposals [B, N_proposals, T, 3]
                - proposal_list: Intermediate proposals from each decoder layer
                - pdm_score: Proposal scoring logits [B, N_proposals]
                - Other prediction outputs
            config: Configuration object with loss weights
        
        Returns:
            loss_dict: Dictionary containing:
                - loss: Total weighted loss
                - Individual loss components for monitoring
                - Evaluation metrics (score, best_score)
        """
        proposals = pred.get("proposals", None)              # [B, N_proposals, T, 3] - final proposals
        proposal_list = pred.get("proposal_list", [])      # List of [B, N_proposals, T, 3] - intermediate proposals
        target_trajectory = targets.get("trajectory", None)      # [B, T, 3] - ground truth

        if proposals is not None:
            # Compute safety and feasibility scores
            final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas = self.compute_score(
                targets, proposals, test=False)

        # Progressive trajectory loss through decoder layers
        trajectory_loss = 0
      
        for proposals_i in proposal_list:
            # Minimum distance loss between proposals and ground truth
            min_loss = torch.linalg.norm(proposals_i - target_trajectory[:, None], dim=-1, ord=1).mean(-1).amin(
                1).mean()  # [B] -> scalar
            
            # Accumulate trajectory loss with decay for earlier layers
            trajectory_loss = config.prev_weight * trajectory_loss + min_loss
            

        # Scoring loss (if available)
        if "pred_logit" in pred.keys():
            sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss = self.score_loss(
                    pred["pred_logit"], 
                    pred["pred_agents_states"], 
                    pred["pred_area_logit"],
                    target_scores, gt_states, gt_valid, gt_ego_areas
                )
        else:
            sub_score_loss = final_score_loss = pred_ce_loss = pred_l1_loss = pred_area_loss = 0

        # BEV semantic segmentation loss (if available)
        if "bev_map_pred" in pred.keys() and pred["bev_map_pred"] is not None:
            bev_semantic_loss = F.cross_entropy(pred["bev_map_pred"], pred["bev_map_gt"].long())
            # bev_semantic_loss = _bev_occ_loss(pred["bev_map_pred"], pred["bev_map_gt"])
        else:
            bev_semantic_loss = 0

        # Denoising alignment loss (if enabled)
        # Uses noise decomposition (inj + sys) and improvement margin loss
        denoising_loss = 0
        denoising_inj_loss = 0
        denoising_nce_pre = 0
        denoising_nce_post = 0
        denoising_imp_loss = 0
        denoising_sys_mag = 0
        
        if 'denoising_loss_total' in pred:
            denoising_loss = pred['denoising_loss_total']
            denoising_inj_loss = pred.get('denoising_loss_inj', pred.get('denoising_loss_noise', 0))
            denoising_nce_pre = pred.get('denoising_loss_nce_pre', 0)
            denoising_nce_post = pred.get('denoising_loss_nce_post', pred.get('denoising_loss_nce', 0))
            denoising_imp_loss = pred.get('denoising_loss_imp', 0)
            denoising_sys_mag = pred.get('denoising_sys_magnitude', 0)
        
        # Get denoising weight from config
        denoise_weight = getattr(config, 'denoise_loss_weight', 1.0)

        # Total weighted loss
        loss = (
                config.trajectory_weight * trajectory_loss        # Trajectory matching loss
                + config.sub_score_weight * sub_score_loss        # Sub scoring loss
                + config.final_score_weight * final_score_loss   # Final scoring loss
                + config.bev_semantic_weight * bev_semantic_loss # BEV segmentation loss
                + denoise_weight * denoising_loss                # Denoising alignment loss
                + config.pred_ce_weight * pred_ce_loss          # Agent existence classification loss
                + config.pred_l1_weight * pred_l1_loss          # Agent state regression loss
                + config.pred_area_weight * pred_area_loss      # Area compliance classification loss
        )

        # Compute evaluation metrics
        if proposals is not None:
            pdm_score = pred["pdm_score"].detach()  # [B, N_proposals] - detached for evaluation
            top_proposals = torch.argmax(pdm_score, dim=1)  # [B] - best proposal indices
            score = final_scores[np.arange(len(final_scores)), top_proposals].mean()  # Average score of top proposals
            best_score = best_scores.mean()  # Average best score across batch
        else:
            score = best_score = 0

        # Package loss components and metrics
        loss_dict = {
            "loss": loss,
            "trajectory_loss": trajectory_loss,
            # "bev_semantic_loss": bev_semantic_loss,
            # 'sub_score_loss': sub_score_loss,
            'final_score_loss': final_score_loss,
            "score": score,
            "best_score": best_score,
            'pred_ce_loss': pred_ce_loss,
            'pred_l1_loss': pred_l1_loss,
            'pred_area_loss': pred_area_loss,
            # Denoising losses for monitoring (noise decomposition + improvement margin)
            "denoising_loss": denoising_loss,
            "denoising_inj_loss": denoising_inj_loss,  # L_inj: delta_inj vs eps_gt
            # "denoising_nce_pre": denoising_nce_pre,    # L_nce_pre: before correction
            "denoising_nce_post": denoising_nce_post,  # L_nce_post: after correction
            "denoising_imp_loss": denoising_imp_loss,  # L_imp: improvement margin
            # "denoising_sys_mag": denoising_sys_mag,    # ||delta_sys||: systematic noise magnitude
        }

        return loss_dict

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
    ) -> Dict:
        """
        Main loss computation interface called during training.
        
        Args:
            features: Input features dictionary (not used directly)
            targets: Ground truth targets dictionary
            pred: Model predictions dictionary
        
        Returns:
            Dictionary of losses and metrics for backpropagation and logging
        """
        return self.alignad_loss(targets, pred, self._config)

    def get_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._alignad_model.parameters(),
            lr=self._lr,
            weight_decay=0.01,
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=30,
            warmup_epochs=1,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        # return torch.optim.Adam(self._alignad_model.parameters(), lr=self._lr)#,weight_decay=1e-4

    def get_coslr_optimizers(self):
        # import ipdb; ipdb.set_trace()
        optimizer_cfg = dict(
            type=self._config.optimizer_type,
            lr=self._lr,
            weight_decay=self._config.weight_decay,
            paramwise_cfg=self._config.opt_paramwise_cfg,
        )
        scheduler_cfg = dict(
            type=self._config.scheduler_type,
            milestones=self._config.lr_steps,
            gamma=0.1,
        )

        optimizer_cfg = DictConfig(optimizer_cfg)
        scheduler_cfg = DictConfig(scheduler_cfg)

        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop("paramwise_cfg", None)

        if paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg["name"]]

            for k, v in self._alignad_model.named_parameters():
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg["name"].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v)
        else:
            params = self._alignad_model.parameters()

        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        # import ipdb; ipdb.set_trace()
        if paramwise_cfg:
            for pg, (_, pg_cfg) in zip(pgs, paramwise_cfg["name"].items()):
                cfg = {}
                if "lr_mult" in pg_cfg:
                    cfg["lr"] = optimizer_cfg["lr"] * pg_cfg["lr_mult"]
                optimizer.add_param_group({"params": pg, **cfg})

        # scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6*1,  # default: 1e-6
            epochs=30,  # default: 20
            warmup_epochs=1, # default: 1
        )

        if "interval" in scheduler_cfg:
            scheduler = {"scheduler": scheduler, "interval": scheduler_cfg["interval"]}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def get_training_callbacks(self):
        """
        Get PyTorch Lightning callbacks for training.
        
        Returns:
            List of callbacks including model checkpointing based on validation score
        """
        checkpoint_cb = ModelCheckpoint(
            save_top_k=20,              # Keep top 20 checkpoints
            monitor='val/score_epoch',  # Monitor validation score
            filename='{epoch}-{step}',  # Checkpoint filename format
            mode="max"                  # Save checkpoints with maximum score
        )
        return [checkpoint_cb]