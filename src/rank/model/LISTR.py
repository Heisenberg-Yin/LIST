from src.rank.model.dualencoder import BiBertEncoder
from src.rank.model.spatial import SpatialModel
import numpy as np
from torch import nn
from torch import Tensor as T
import torch
from scipy.spatial import distance
import torch.nn.functional as F
import math
import math
import torch.nn.init as init


def dot_product_scores(q_vectors, ctx_vectors):
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r
    
class LISTR(nn.Module):
    def __init__(self, args):
        super(LISTR, self).__init__()
        self.biencoder = BiBertEncoder(args)
        self.spatial_model = SpatialModel(args)

        self.text_aware_attention = nn.Linear(768, 2)

        self.device = args.device
        
        if args.dataset == "beijing":
            self.max_distance = 227952.22531887464 
        elif args.dataset == "geo-glue":
            self.max_distance = 598971.3028776475
        elif args.dataset == "shanghai":
            self.max_distance = 203746.56912588517
        else:
            raise ValueError("dataset not implemented")

    
    def get_final_score(self, query_vectors: T, poi_vectors: T, distance_score: T):
        text_score = dot_product_scores(query_vectors, poi_vectors)
        attention = self.text_aware_attention(query_vectors)
        scores = []
        for i in range(attention.size()[0]):
            scores.append((attention[i][0]*text_score[i,:]+attention[i][1]*distance_score[i,:]).view(1,-1))
        scores = torch.cat(scores, dim=0)
        
        return scores
    
    
       
    def get_attention_final_score(self, poi_ids: T, poi_segments: T, poi_attn_mask: T, poi_coordinate: np.array, query_ids: T, query_segments: T, query_attn_mask: T, query_coordinate: np.array):
        poi_emb = self.biencoder.body_emb(input_ids = poi_ids, token_type_ids = poi_segments, attention_mask = poi_attn_mask)
        query_emb = self.biencoder.query_emb(input_ids = query_ids, token_type_ids = query_segments, attention_mask = query_attn_mask)        
        candidates_distance = distance.cdist(query_coordinate, poi_coordinate)
        candidates_distance = 1 - candidates_distance / self.max_distance
        spatial_score = self.spatial_model(T(candidates_distance).to(next(self.spatial_model.parameters()).device))
        text_score = dot_product_scores(query_emb, poi_emb)        
        attention = self.text_aware_attention(query_emb)
        
        scores = []
        for i in range(attention.size()[0]):
            scores.append((attention[i][0]*text_score[i,:]+attention[i][1]*spatial_score[i,:]).view(1,-1))
        return torch.cat(scores, dim=0)
    
    def get_poi_representation(self, poi_ids: T, poi_segments: T, poi_attn_mask: T) -> T:
        poi_emb = self.biencoder.body_emb(input_ids = poi_ids, token_type_ids = poi_segments, attention_mask = poi_attn_mask)
        return poi_emb
    
    def get_query_representation(self, query_ids: T, query_segments: T, query_attn_mask: T) -> T:
        query_emb = self.biencoder.query_emb(input_ids = query_ids, token_type_ids = query_segments, attention_mask = query_attn_mask)
        return query_emb
    
    def get_representation(self, poi_ids: T, poi_segments: T, poi_attn_mask: T, query_ids: T, query_segments: T, query_attn_mask: T):
        poi_emb = self.get_poi_representation(poi_ids=poi_ids, poi_segments=poi_segments, poi_attn_mask=poi_attn_mask)
        query_emb = self.get_query_representation(query_ids=query_ids, query_segments=query_segments, query_attn_mask=query_attn_mask)
        return poi_emb, query_emb
    
    def get_spatial_score(self, query_coordinate: np.array, poi_coordinate: np.array):
        candidates_distance = distance.cdist(query_coordinate, poi_coordinate)
        candidates_distance = 1 - candidates_distance / self.max_distance
        spatial_score = self.spatial_model(T(candidates_distance).to(next(self.spatial_model.parameters()).device))
        return spatial_score

    def get_attention_score(self, query_emb):
        attention = self.text_aware_attention(query_emb)
        return attention

