import torch
import torch.nn as nn
from ..encoder.transformer_decoder import MLP
from ..alignad_config import AlignADConfig

class Scorer(nn.Module):
    def __init__(self, config: AlignADConfig):
        super().__init__()

        self.b2d = config.b2d

        self.proposal_num = config.proposal_num
        self.score_num = config.score_num

        self.pred_score = MLP(config.tf_d_model, config.tf_d_ffn, self.score_num)

        # agent prediction and mapping
        self.agent_pred= config.agent_pred
        if self.agent_pred:
            if self.b2d:
                self.pred_col_agent = MLP(config.tf_d_model, config.tf_d_ffn, 2*6* 9)
            else:
                self.pred_col_agent = MLP(config.tf_d_model, config.tf_d_ffn,2* 40 * 9)

        self.area_pred=config.area_pred
        if self.area_pred:
            if self.b2d:
                self.pred_area =  MLP(config.tf_d_model, config.tf_d_ffn, 2)
            else:
                self.pred_area =  MLP(config.tf_d_model, config.tf_d_ffn, 5*2)

    def forward(self, proposals, bev_feature):
        batch_size=len(proposals)   # bs=proposals.shape[0]
        p_size=proposals.shape[1]   # 64
        t_size=proposals.shape[2]   # 8

        proposal_feature = bev_feature.reshape(batch_size, p_size, t_size, -1).amax(-2) # b, 64, 512
        pred_logit = self.pred_score(proposal_feature).reshape(batch_size, -1, self.score_num)  # b, 64, 6

        pred_agents_states = pred_area_logit = None

        if self.training:
            if self.area_pred:
                pred_area_logit = self.pred_area(bev_feature)

            if self.agent_pred:
                pred_agents_states = self.pred_col_agent(proposal_feature).reshape(batch_size,p_size,t_size,-1,2,9)

        return pred_logit, pred_agents_states, pred_area_logit
