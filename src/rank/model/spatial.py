from torch import nn
from torch import Tensor as T
import torch
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

class SpatialModel(nn.Module):
    """ 
    Bi-Encoder model component. 
    Encapsulates query context and poi context encoders.
    """
    def __init__(self, args):
        super(SpatialModel, self).__init__()
        
        self.spatial_step_k = args.spatial_step_k
        self.spatial_dropout = args.spatial_dropout

        self.dropout = nn.Dropout(p=args.spatial_dropout)

       
        self.weight = nn.Parameter(torch.Tensor(self.spatial_step_k, 1))
        self.weight_exp = nn.Parameter(torch.Tensor(2, 1))
        self.reset_parameters()
        init.xavier_uniform_(self.weight)

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.weight, gain=gain)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            gain = nn.init.calculate_gain('relu')
            nn.init.xavier_uniform_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, distance_scores: T) -> T:
        t = torch.linspace(0, 1, self.spatial_step_k+1)[1:].to(distance_scores.device)
        t_step = (distance_scores.view(-1,1) > t).float()

        t_score = F.relu(self.weight)            
        score = torch.mm(t_step.float().to(t_score.device), t_score)
        score = torch.sum(score, dim=1).view(distance_scores.size())
        return score
     