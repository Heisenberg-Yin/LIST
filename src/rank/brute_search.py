import torch
from src.rank.model.LISTR import LISTR
import logging
from src.utils import load_states_from_checkpoint, set_seed, is_first_worker
import numpy as np
import os
from transformers import BertTokenizer
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from tqdm import tqdm
import pickle
import pandas as pd
from collections import defaultdict
from torchmetrics.functional import retrieval_normalized_dcg
import json
import argparse
from scipy.spatial import distance
import torch.nn.functional as F

def custom_ndcg(prediction, truth, k):
    # prediction: list of indices
    # # truth: list of indices
    # # k: int
    # # return: float
    dcg = 0
    for i in range(min(len(prediction), k)):
        if prediction[i] in truth:
            dcg += 1 / np.log2(i + 2)
    idcg = 0
    for i in range(min(len(truth), k)):
        idcg += 1 / np.log2(i + 2)
    return dcg / idcg

logger = logging.getLogger(__name__)


def load_rel(rel_path):
    reldict = defaultdict(list)
    rel_df = pd.read_csv(rel_path, sep=',')
    for index, row in rel_df.iterrows():
        qid, pid = row['query_id'], row['poi_id']
        reldict[qid].append(pid)
    return dict(reldict)


def load_model(args):

    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # args.model_type = args.model_type
   
    if is_first_worker():
        # Create output directory if needed
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
    
    tokenizer = BertTokenizer.from_pretrained(args.model_type)
    
    model = LISTR(args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    return tokenizer, model

def set_env(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(f"cuda:{args.device}" if not args.no_cuda else "cpu")
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    

class geo_object():
    def __init__(self, id, embedding: np.array, coordinate: List[object]):
        self.id = id
        self.embedding = embedding
        self.coordinate = coordinate

class QueryDataset(Dataset):
    def __init__(self, queries):
        self.queries = queries
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, index):
        query = self.queries[index]
        query_id = query.id
        embedding = query.embedding
        coordinate = query.coordinate
        return query_id, embedding, coordinate
                
    @classmethod
    def get_collate_fn(cls):
        def create_biencoder_input2(features):
            id_list = []            
            embed_list = []
            coor_list = []

            for index, feature in enumerate(features):
                id_list.append(feature[0])               
                embed_list.append(feature[1])
                coor_list.append(feature[2])

            
            return  id_list, embed_list, coor_list
        return create_biencoder_input2

def main(args):
    qrels_test = load_rel(args.origin_data_dir+"qrels_test.csv")
    tokenizer, model = load_model(args)
    basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(basic_format)
    log_path = os.path.join(args.log_dir, 'log.txt')
    handler = logging.FileHandler(log_path, 'a', 'utf-8')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    
    checkpoint_path = args.checkpoint_file

    saved_state = load_states_from_checkpoint(checkpoint_path)
    model.load_state_dict(saved_state.model_dict,strict=False)
    model.eval()
    poi_embedding_file = os.path.join(args.embedding_dir, 'poi_embedding.pkl')
    
    with open(poi_embedding_file, "rb") as f:
        poi_embedding_matrix = pickle.load(f)    
    
    query_embedding_file = os.path.join(args.embedding_dir, 'query_embedding.pkl')

    with open(query_embedding_file, "rb") as f:
        query_embedding_matrix = pickle.load(f)

    POIs = []
    Querys = []

    poi_df = pd.read_csv(args.origin_data_dir + "poi.csv", sep=',')
    for index, row in poi_df.iterrows():
        poi_id = int(row.id)
        POIs.append(geo_object(poi_id, poi_embedding_matrix[poi_id][1],eval(row.coor)))
    
    query_df = pd.read_csv(args.origin_data_dir + "query_test.csv", sep=',')
    for index, row in query_df.iterrows():
        query_id = int(row.id)
        Querys.append(geo_object(query_id, query_embedding_matrix[query_id][1], eval(row.coor)))       
    
    query_dataset = QueryDataset(Querys)
    query_sample = SequentialSampler(query_dataset) 

    query_dataloader = DataLoader(query_dataset, sampler=query_sample,
                        batch_size=32, collate_fn = QueryDataset.get_collate_fn())    
    embedding_poi_matrix_for_rank = torch.tensor(np.concatenate([embeddings[1].reshape(1,-1) for embeddings in poi_embedding_matrix], axis=0))
    
    poi_coordinates = [poi.coordinate for poi in POIs]  
    poi_coordinates = np.concatenate([np.array(coor).reshape(1, -1) for coor in poi_coordinates], axis=0)

    recalls_100 = []
    recalls_50 = []
    recalls_20 = []
    recalls_10 = []
        
    ndcgs_1 = []
    ndcgs_3 = []
    ndcgs_5 = []
    ndcgs_10 = []
    spatial_Weight = F.relu(model.spatial_model.weight).view(-1).tolist()
    for i in range(1, len(spatial_Weight)):
        spatial_Weight[i] = spatial_Weight[i] + spatial_Weight[i-1]
    spatial_Weight = [0] + spatial_Weight
    spatial_Weight = [float(x) for x in spatial_Weight]
    spatial_Weight = np.array(spatial_Weight).astype(float)
    spatial_Weight = torch.Tensor(spatial_Weight).to(args.device)
    logger.info(spatial_Weight.size())
    # max_distance = model.max_distance
    if args.dataset == "beijing":
        max_distance = 227952.22531887464 
    elif args.dataset == "geo-glue":
        max_distance = 598971.3028776475
    elif args.dataset == "shanghai":
        max_distance = 203746.56912588517    
    spatial_step = args.spatial_step
    embedding_poi_matrix_for_rank = embedding_poi_matrix_for_rank.to(args.device)
    for i, batch in tqdm(enumerate(query_dataloader), total=len(query_dataloader)):
        query_ids = batch[0]
        query_embeds = batch[1]
        query_coordinates = batch[2]
        query_coordinates = np.concatenate([np.array(coor).reshape(1, -1) for coor in query_coordinates], axis=0)
        query_embedding = torch.Tensor(np.concatenate([emb.reshape(1, -1) for emb in query_embeds], axis=0)).to(args.device)
        

        
        candidates_distance = distance.cdist(query_coordinates, poi_coordinates)
        candidates_distance = 1 - candidates_distance / max_distance
        spatial_idx = np.floor(candidates_distance/spatial_step).astype(int)
        spatial_score = spatial_Weight[spatial_idx.reshape(-1)].view(np.shape(spatial_idx)[0], -1)
        
        scores = model.get_final_score(query_embedding, embedding_poi_matrix_for_rank, spatial_score).cpu()
        
        
        top_indices = torch.topk(scores, 100, largest=True)[1].cpu().numpy()
        top_indices = top_indices.tolist()

        for j in range(len(top_indices)):
            predict_id = top_indices[j]

            truth = np.array(qrels_test[query_ids[j]])            
            recall_100 = len(set(predict_id[:100]) & set(truth)) / min(len(truth),100)
            recall_50 = len(set(predict_id[:50]) & set(truth)) / min(len(truth),50)
            recall_20 = len(set(predict_id[:20]) & set(truth)) / min(len(truth),20)
            recall_10 = len(set(predict_id[:10]) & set(truth)) / min(len(truth),10)
            
            recalls_100.append(recall_100)
            recalls_50.append(recall_50)
            recalls_20.append(recall_20)
            recalls_10.append(recall_10)
            
            ndcg_1 = custom_ndcg(predict_id,truth.tolist(), 1)
            ndcg_3 = custom_ndcg(predict_id,truth.tolist(), 3)
            ndcg_5 = custom_ndcg(predict_id,truth.tolist(), 5)
            ndcg_10 = custom_ndcg(predict_id,truth.tolist(), 10)

            
            ndcgs_1.append(ndcg_1)
            ndcgs_3.append(ndcg_3)
            ndcgs_5.append(ndcg_5)
            ndcgs_10.append(ndcg_10)

    print('recall@100', np.mean(recalls_100))
    print('recall@50', np.mean(recalls_50))
    print('recall@20', np.mean(recalls_20))
    print('recall@10', np.mean(recalls_10))

    print('ndcg@1', np.mean(ndcgs_1))
    print('ndcg@3', np.mean(ndcgs_3))
    print('ndcg@5', np.mean(ndcgs_5))
    print('ndcg@10', np.mean(ndcgs_10))      

def run_parse_args():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--checkpoint_file", type=str)
    parser.add_argument("--embedding_dir", type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--model_type", default=None, type=str, help="config name for model initialization")
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_poi_length", type=int, default=96)
    parser.add_argument("--dataset", default="beijing", type=str, help="dataset to be used")    
    
    parser.add_argument(
        "--no_cuda", 
        action="store_true", 
        help="Avoid using CUDA when available"
    )
        
    parser.add_argument(
        "--per_gpu_batch_size", 
        default=32, 
        type=int, 
        help="Batch size per GPU/CPU for training.",
    ) 
    parser.add_argument(
        "--n_heads", 
        default=3, 
        type=int, 
        help="Batch size per GPU/CPU for training.",
    )     
    parser.add_argument(
            "--origin_data_dir",
            default=None,
            type=str,
    )
    parser.add_argument('--share_weight', action='store_true') 
    parser.add_argument('--bert_dropout', type=float, default=0)
    parser.add_argument('--att_dropout', type=float, default=0)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument('--spatial_dropout', type=float, default=0)
    parser.add_argument('--spatial_step_k', type=int, default=1000)
    parser.add_argument('--spatial_step', type=float, default=0.001)
    parser.add_argument(
            "--gradient_checkpointing",
            default=False,
            action="store_true",
    )     
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--topn', type=int, default=100)
    parser.add_argument('--fp16', action='store_true')    
    args = parser.parse_args()

    if is_first_worker():
        # Create output directory if needed
        if not os.path.exists(os.path.join(args.log_dir,args.dataset)):
            os.makedirs(os.path.join(args.log_dir,args.dataset))
        if not os.path.exists(os.path.join(args.output_dir,args.dataset)):
            os.makedirs(os.path.join(args.output_dir,args.dataset))        
    args.log_dir = os.path.join(args.log_dir,args.dataset)
    args.output_dir = os.path.join(args.output_dir,args.dataset)

    return args

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = run_parse_args() 
    set_env(args)    
    # setup_args_gpu(args)
    main(args)
    