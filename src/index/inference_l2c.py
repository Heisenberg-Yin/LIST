import torch
# import torch.nn as nn
from src.rank.model.spatialdual import DualSpatial
import logging
import argparse
# from src.inference_dual_encoder_args import run_parse_args
from src.utils import load_states_from_checkpoint, set_seed, is_first_worker
import numpy as np
import os
# from transformers import BertTokenizer
import torch.nn.functional as F
import numpy as np
# import matplotlib.pyplot as plt
import time
from typing import Tuple, List
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import pathlib
import pickle
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict
from typing import Tuple, List
# from torchmetrics.functional import retrieval_normalized_dcg
import csv
from src.cluster.l2c import FeedForwardNet
from scipy.spatial import distance

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
    # Prepare GLUE task
    args.output_mode = "classification"
    label_list = ["0", "1"]
    num_labels = len(label_list)

    # store args
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
    
    # tokenizer = BertTokenizer.from_pretrained(args.model_type)
    
    model = DualSpatial(args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    return model


def run_parse_args():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--bert_dropout', type=float, default=0)
    parser.add_argument("--model_type", default=None, type=str, help="config name for model initialization")
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--embedding_dir", type=str)
    parser.add_argument("--cluster_checkpoint_file", type=str)
    parser.add_argument("--rank_checkpoint_file", type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--min_cluster_size", type=int, default=1000)
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_poi_length", type=int, default=96)
    parser.add_argument("--dataset", default="beijing", type=str, help="dataset to be used")    
    parser.add_argument('--spatial_step', type=float, default=0.001)
    parser.add_argument(
        "--no_cuda", 
        action="store_true", 
        help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", 
        default=8, 
        type=int, 
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
            "--hidden_dim",
            default=256,
            type=int,
    )
    parser.add_argument(
            "--num_cluster",
            default=10,
            type=int,
    )           
    parser.add_argument(
            "--num_layers",
            default=2,
            type=int,
    )               
    parser.add_argument(
            "--dropout",
            default=0,
            type=float,
    )  
    parser.add_argument(
            "--att_dropout",
            default=0,
            type=float,
    )     
    parser.add_argument(
            "--n_heads",
            default=2,
            type=int,
    )    
    parser.add_argument(
        "--per_gpu_batch_size", 
        default=32, 
        type=int, 
        help="Batch size per GPU/CPU for training.",
    ) 
    parser.add_argument(
            "--origin_data_dir",
            default=None,
            type=str,
    )
    parser.add_argument(
            "--gradient_checkpointing",
            default=False,
            action="store_true",
    )        
  
    parser.add_argument('--spatial_dropout', type=float, default=0)
    parser.add_argument('--spatial_step_k', type=int, default=1000)
    parser.add_argument('--spatial_hidden', type=int, default=128)
    # parser.add_argument('--use_attention', action='store_true')
    parser.add_argument('--min_precision', type=float, default=0.9)
    # parser.add_argument('--latitude_bins', type=int, default=1000)
    # parser.add_argument('--geo_embedding_dim', type=int, default=256)    
    # parser.add_argument('--spatial_use_mlp', action='store_true') 
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--topn', type=int, default=100)
    parser.add_argument('--fp16', action='store_true')    
    args = parser.parse_args()
    return args

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

class L2C_inference_Dataset(Dataset):
    def __init__(self, poi_data):
        self.poi_data = poi_data
        
    def __len__(self):
        return len(self.poi_data)

    def __getitem__(self, idx):
        id = self.poi_data[idx].id
        coordinates = self.poi_data[idx].coordinate
        embedding = self.poi_data[idx].embedding
        return {
                'id': id,
                'embedding': embedding,
                'coordinates': coordinates,
                }
    
    @classmethod
    def get_collate_fn(cls):
        def create_input(batch):
            ids = [item['id'] for item in batch]
            embs = [item['embedding'] for item in batch]
            coors = [item['coordinates'] for item in batch]
            return ids, coors, embs
        return create_input

def adjust_clusters(cluster_size, precisions, cluster_inverted_id, sorted_cluster_ids, items, min_precision=0.8):
    for cluster_id, _ in enumerate(cluster_size):
        if precisions[cluster_id] < min_precision:

            item_list = cluster_inverted_id[cluster_id].copy()
            for item_id in item_list:
                sorted_cluster = sorted_cluster_ids[item_id]              
                for sorted_cluster_id in sorted_cluster:
                    if precisions[sorted_cluster_id] < min_precision:
                        continue
                    else:
                        cluster_inverted_id[cluster_id].remove(item_id)
                        cluster_size[cluster_id] -= 1                        
                        assert item_id not in cluster_inverted_id[sorted_cluster_id]
                        items[item_id].cluster_id = sorted_cluster_id
                        cluster_inverted_id[sorted_cluster_id].append(item_id)
                        cluster_size[sorted_cluster_id] += 1
                        break
    return cluster_inverted_id, cluster_size

def adjust_precision(query_to_cluster_inverted_id, poi_to_cluster_inverted_id, test_Querys, qrels_test):
    precisions = []    
    for cluster_id, query_ids in enumerate(query_to_cluster_inverted_id):
        # query_ids = query_to_cluster_inverted_id[cluster_id]
        poi_ids = poi_to_cluster_inverted_id[cluster_id]
        logger.info("query set size is {}".format(len(query_ids)))
        logger.info("poi set size is {}".format(len(poi_ids)))
        
        precision = []
        for query_id in query_ids:
            truth = qrels_test[test_Querys[query_id].id]
            precision.append(len(set(poi_ids) & set(truth))/ len(truth))        
        if len(precision) != 0:
            precision = np.mean(precision)
        else:
            precision = 0
        precisions.append(precision)
        logger.info(f"Average precision in cluster {cluster_id}: {precision}") 
    return precisions

class geo_object:
    def __init__(self, poi_id, poi_embedding, coordinate: List[object]):
        self.id = poi_id
        self.embedding = torch.Tensor(poi_embedding).view(1,-1)
        self.coordinate = coordinate
        self.cluster_id = -1

def read_query_file(file_path):
    json_list = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            item = dict()
            query_id = int(row[0])
            coordinate = [float(row[1]), float(row[2])]
            attention = [float(row[3]), float(row[4])]
            embedding = [float(x) for x in row[5].split()]
            item['id'] = query_id
            item['coordinate'] = coordinate
            item['attention'] = attention
            item['embedding'] = embedding
            json_list.append(item)
    return json_list

def read_poi_file(file_path):
    json_list = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            item = dict()
            poi_id = int(row[0])
            coordinate = [float(row[1]), float(row[2])]
            embedding = [float(x) for x in row[3].split()]
            item['id'] = poi_id
            item['coordinate'] = coordinate
            item['embedding'] = embedding
            json_list.append(item)
    return json_list

def write_query_to_file(query, file):
    docstring = "{},{},{},{},{},".format(query['id'], query['coordinate'][0], query['coordinate'][1], query['attention'][0], query['attention'][1])
    query_emb = query['embedding']
    for j in range(len(query_emb)):
        docstring = docstring + str(query_emb[j])
        if j != len(query_emb) - 1 :
            docstring = docstring + " "
    docstring = docstring + "\n"            
    file.write(docstring)

def write_poi_to_file(poi, file):
    docstring = "{},{},{},".format(poi['id'], poi['coordinate'][0], poi['coordinate'][1])
    poi_emb = poi['embedding']
    for j in range(len(poi_emb)):
        docstring = docstring + str(poi_emb[j])
        if j != len(poi_emb) - 1 :
            docstring = docstring + " "
    docstring = docstring + "\n"            
    file.write(docstring)    

# Replace with your file path

def main():
    args = run_parse_args() 
    queries_after_inference = read_query_file(os.path.join(args.origin_data_dir,'query_after_inference.csv'))
    print("len of queries_after_inference",len(queries_after_inference)) 
    pois_after_inference = read_poi_file(os.path.join(args.origin_data_dir,'poi_after_inference.csv'))
    print("len of pois_after_inference",len(pois_after_inference)) 
    # exit(1)    
    qrels_test = load_rel(args.origin_data_dir + "qrels_test.csv")
    qrels_val = load_rel(args.origin_data_dir + "qrels_val.csv")
    set_env(args)    
    basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(basic_format)
    log_path = os.path.join(args.log_dir, 'log.txt')
    handler = logging.FileHandler(log_path, 'a', 'utf-8')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    
    cluster_checkpoint_path = args.cluster_checkpoint_file
    saved_state = load_states_from_checkpoint(cluster_checkpoint_path)
    model = FeedForwardNet(in_feats=770, hidden=args.hidden_dim, out_feats=args.num_cluster, n_layers=args.num_layers, dropout=args.dropout, dataset=args.dataset).to(args.device) 
    model.load_state_dict(saved_state.model_dict,strict=False)
    model.eval()
    poi_embedding_file = args.embedding_dir + 'poi_embedding.pkl'
    query_embedding_file = args.embedding_dir + 'query_embedding.pkl'
    
    with open(poi_embedding_file, "rb") as f:
        poi_embedding_matrix = pickle.load(f)    
    
    with open(query_embedding_file, "rb") as f:
        query_embedding_matrix = pickle.load(f)
    
    POIs = []
    poi_df = pd.read_csv(args.origin_data_dir + "poi.csv", sep=',')
    dXMin = 1e9      
    dXMax = 0       
    dYMin = 1e9       
    dYMax = 0   
    for index, row in poi_df.iterrows():
        poi_id = int(row.id)
        POIs.append(geo_object(poi_id, poi_embedding_matrix[poi_id][1], eval(row.coor)))
        dXMin = min(POIs[-1].coordinate[0], dXMin)
        dYMin = min(POIs[-1].coordinate[1], dYMin)
        dXMax = max(POIs[-1].coordinate[0], dXMax)
        dYMax = max(POIs[-1].coordinate[1], dYMax)
    
    logger.info("dXMin is {}, dYMin is {}, dXMax is {}, dYMax is {}".format(dXMin, dYMin, dXMax, dYMax))
    
    poi_dataset = L2C_inference_Dataset(POIs)
    poi_sampler = SequentialSampler(poi_dataset)
    poi_dataloader = DataLoader(poi_dataset, sampler=poi_sampler,
                        collate_fn=L2C_inference_Dataset.get_collate_fn(),
                        batch_size=args.per_gpu_eval_batch_size)
    
    poi_to_cluster_inverted_id = [[] for _ in range(args.num_cluster)]
    poi_to_cluster_size = [0]*args.num_cluster
    poi_to_cluster_sorted_cluster_ids = {}
    for _, (poi_id, coordinates, embedding) in tqdm(enumerate(poi_dataloader)):
        geo_embed = model.get_geo_representation(coordinates).to(args.device)
        text_embed = torch.cat(embedding, dim=0).to(args.device)
        input_embed = torch.cat([text_embed, geo_embed], dim=1)
        # input_embed = text_embed
        with torch.no_grad():
            logits = model(input_embed)
            prob = F.softmax(logits,dim=1)
            cluser_ids = torch.max(logits, dim=1)[1]
            sorted_probs, sorted_cluster_ids = torch.sort(prob, dim=1, descending=True)
            for i, cluster_id, sorted_cluster_id in zip(poi_id, cluser_ids.tolist(), sorted_cluster_ids.tolist()):
                POIs[i].cluster_id = cluster_id
                poi_to_cluster_inverted_id[cluster_id].append(i)
                poi_to_cluster_size[cluster_id] +=1
                poi_to_cluster_sorted_cluster_ids[i] = sorted_cluster_id
    logger.info(poi_to_cluster_size)    
    logger.info([len(cluster) for cluster in poi_to_cluster_inverted_id])
    
    
    tot = 0
    uf = 0
    for i in range(len(poi_to_cluster_inverted_id)):
        tot += len(poi_to_cluster_inverted_id[i])
        uf += len(poi_to_cluster_inverted_id[i]) * len(poi_to_cluster_inverted_id[i])
    
    uf = uf * args.num_cluster / (tot * tot)
    
    print("uf is ", uf)
    # exit(1)
        
    test_Querys = []
    query_test_df = pd.read_csv(args.origin_data_dir + "query_test.csv", sep=',')
    for index, row in query_test_df.iterrows():
        query_id = int(row.id)
        test_Querys.append(geo_object(query_id, query_embedding_matrix[query_id][1], eval(row.coor)))     

    val_Querys = []
    query_val_df = pd.read_csv(args.origin_data_dir + "query_val.csv", sep=',')
    for index, row in query_val_df.iterrows():
        query_id = int(row.id)
        val_Querys.append(geo_object(query_id, query_embedding_matrix[query_id][1], eval(row.coor)))     
    
    test_query_dataset = L2C_inference_Dataset(test_Querys)
    test_query_sampler = SequentialSampler(test_query_dataset)
    test_query_dataloader = DataLoader(test_query_dataset, sampler=test_query_sampler,
                        collate_fn=L2C_inference_Dataset.get_collate_fn(),
                        batch_size=args.per_gpu_eval_batch_size)
    
    val_query_dataset = L2C_inference_Dataset(val_Querys)
    val_query_sampler = SequentialSampler(val_query_dataset)
    val_query_dataloader = DataLoader(val_query_dataset, sampler=val_query_sampler,
                        collate_fn=L2C_inference_Dataset.get_collate_fn(),
                        batch_size=args.per_gpu_eval_batch_size)
    
    
    
    test_query_to_cluster_inverted_id = [[] for _ in range(args.num_cluster)]
    test_query_to_cluster_size = [0]*args.num_cluster
    test_query_to_cluster_sorted_cluster_ids = []
    test_query_count = 0 
    for _, (query_ids, coordinates, embedding) in tqdm(enumerate(test_query_dataloader)):
        geo_embed = model.get_geo_representation(coordinates).to(args.device)
        text_embed = torch.cat(embedding, dim=0).to(args.device)
        input_embed = torch.cat([text_embed, geo_embed], dim=1)
        # input_embed = text_embed
        with torch.no_grad():
            logits = model(input_embed)
            prob = F.softmax(logits,dim=1)
            cluster_ids = torch.max(prob, dim=1)[1]
            sorted_probs, sorted_cluster_ids = torch.sort(prob, dim=1, descending=True)
            test_query_to_cluster_sorted_cluster_ids.extend(sorted_cluster_ids.tolist())
            for cluster_id in cluster_ids.tolist():
                test_Querys[test_query_count].cluster_id = cluster_id
                test_query_to_cluster_inverted_id[cluster_id].append(test_query_count)
                test_query_count += 1
                test_query_to_cluster_size[cluster_id] += 1  
    logger.info(test_query_to_cluster_size)
    
    
    
        
    val_query_to_cluster_inverted_id = [[] for _ in range(args.num_cluster)]
    val_query_to_cluster_size = [0]*args.num_cluster
    val_query_to_cluster_sorted_cluster_ids = []
    val_query_count = 0 
    for _, (query_ids, coordinates, embedding) in tqdm(enumerate(val_query_dataloader)):
        geo_embed = model.get_geo_representation(coordinates).to(args.device)
        text_embed = torch.cat(embedding, dim=0).to(args.device)
        input_embed = torch.cat([text_embed, geo_embed], dim=1)
        # input_embed = text_embed
        with torch.no_grad():
            logits = model(input_embed)
            prob = F.softmax(logits,dim=1)
            cluster_ids = torch.max(prob, dim=1)[1]
            sorted_probs, sorted_cluster_ids = torch.sort(prob, dim=1, descending=True)
            val_query_to_cluster_sorted_cluster_ids.extend(sorted_cluster_ids.tolist())
            for cluster_id in cluster_ids.tolist():
                val_Querys[val_query_count].cluster_id = cluster_id
                val_query_to_cluster_inverted_id[cluster_id].append(val_query_count)
                val_query_count += 1
                val_query_to_cluster_size[cluster_id] += 1  
    logger.info(val_query_to_cluster_size)
    
    precisions = adjust_precision(val_query_to_cluster_inverted_id, poi_to_cluster_inverted_id, val_Querys, qrels_val)
    average_precision = 0 
    tot = 0
    for i, precision in enumerate(precisions):
        average_precision = average_precision + precision * len(val_query_to_cluster_inverted_id[i])
        tot = tot + len(val_query_to_cluster_inverted_id[i])
        if precision != 0 and precision < args.min_precision:
            flag=False
            break
    print("average_precision: ", average_precision/tot)
    
    while True:
        poi_to_cluster_inverted_id, poi_to_cluster_size = adjust_clusters(poi_to_cluster_size, precisions, poi_to_cluster_inverted_id, poi_to_cluster_sorted_cluster_ids, POIs, min_precision = args.min_precision)
        val_query_to_cluster_inverted_id, val_query_to_cluster_size = adjust_clusters(val_query_to_cluster_size, precisions, val_query_to_cluster_inverted_id, val_query_to_cluster_sorted_cluster_ids, val_Querys, min_precision = args.min_precision)
        test_query_to_cluster_inverted_id, test_query_to_cluster_size = adjust_clusters(test_query_to_cluster_size, precisions, test_query_to_cluster_inverted_id, test_query_to_cluster_sorted_cluster_ids, test_Querys, min_precision = args.min_precision)
        
        precisions = adjust_precision(val_query_to_cluster_inverted_id, poi_to_cluster_inverted_id, val_Querys, qrels_val)
        # If all precisions are good (>= 0.8), then break the while loop
        flag = True
        average_precision = 0 
        tot = 0
        for i, precision in enumerate(precisions):
            average_precision = average_precision + precision * len(val_query_to_cluster_inverted_id[i])
            tot = tot + len(val_query_to_cluster_inverted_id[i])
            if precision != 0 and precision < args.min_precision:
                flag=False
                break
        print("average_precision: ", average_precision/tot)
        if flag:
            break
    
    
    
    logger.info(poi_to_cluster_size)    
    logger.info([len(cluster) for cluster in poi_to_cluster_inverted_id])                    
    logger.info(val_query_to_cluster_size)
    logger.info([len(cluster) for cluster in val_query_to_cluster_inverted_id])
    logger.info(test_query_to_cluster_size)
    logger.info([len(cluster) for cluster in test_query_to_cluster_inverted_id])    
    
    
    total = 0
    for query_number, poi_number in zip(test_query_to_cluster_size, poi_to_cluster_size):
        total = total + query_number*poi_number

    print(total/sum(test_query_to_cluster_size))
    
    tot = 0
    uf = 0
    for i in range(len(poi_to_cluster_inverted_id)):
        tot += len(poi_to_cluster_inverted_id[i])
        uf += len(poi_to_cluster_inverted_id[i]) * len(poi_to_cluster_inverted_id[i])
    
    uf = uf * args.num_cluster / (tot * tot)
    
    print("uf is ", uf)    
    
    
    
    
    rank_model = load_model(args)
    checkpoint_path = args.rank_checkpoint_file
    saved_state = load_states_from_checkpoint(checkpoint_path)
    rank_model.load_state_dict(saved_state.model_dict,strict=False)
    rank_model.eval()
    recalls_100 = []
    recalls_50 = []
    recalls_20 = []
    recalls_10 = []
    ndcgs_1 = []
    ndcgs_3 = []
    ndcgs_5 = []
    ndcgs_10 = []    
    spatial_Weight = F.relu(rank_model.spatial_model.weight).view(-1).tolist()
    for i in range(1, len(spatial_Weight)):
        spatial_Weight[i] = spatial_Weight[i] + spatial_Weight[i-1]
    spatial_Weight = [0] + spatial_Weight
    spatial_Weight = [float(x) for x in spatial_Weight]
    spatial_Weight = np.array(spatial_Weight).astype(float)
    spatial_Weight = torch.Tensor(spatial_Weight).to(args.device)
    max_distance = rank_model.max_distance
    spatial_step = args.spatial_step
        
    with torch.no_grad():
        for cluster_id in range(len(test_query_to_cluster_inverted_id)):
            partital_ndcgs_1, partital_ndcgs_3, partital_ndcgs_5, partital_ndcgs_10 = [], [], [], []
            partital_recalls_100, partital_recalls_50, partital_recalls_20, partital_recalls_10 = [], [], [], []            
            query_ids = test_query_to_cluster_inverted_id[cluster_id]
            poi_ids = poi_to_cluster_inverted_id[cluster_id]
            
            if(len(query_ids) == 0):
                continue
            else:
                logger.info("query set size is {}".format(len(query_ids)))
                logger.info("poi set size is {}".format(len(poi_ids)))
            
            precision = []
            for query_id in query_ids:
                truth = qrels_test[test_Querys[query_id].id]
                precision.append(len(set(poi_ids) & set(truth))/ len(truth))
            
            logger.info(f"Average precision in cluster {cluster_id}: {np.mean(precision)}")  
            poi_emb_tensor = torch.cat([POIs[poi_id].embedding for poi_id in poi_ids], dim=0).to(args.device)
            poi_coordinates = np.concatenate([np.array(POIs[poi_id].coordinate).reshape(1,-1) for poi_id in poi_ids], axis=0)             
            query_id_batches = [query_ids[i:i + args.per_gpu_eval_batch_size] for i in range(0, len(query_ids), args.per_gpu_eval_batch_size)]
            for batch_query_ids in query_id_batches:
                query_emb_tensor = torch.cat([test_Querys[query_id].embedding for query_id in batch_query_ids], dim=0).to(args.device)
                query_coordinates = np.concatenate([np.array(test_Querys[query_id].coordinate).reshape(1,-1) for query_id in batch_query_ids], axis=0)

                # spatial_score = rank_model.get_spatial_score(query_coordinates, poi_coordinates)
                candidates_distance = distance.cdist(query_coordinates, poi_coordinates)
                candidates_distance = 1 - candidates_distance / max_distance
                spatial_idx = np.floor(candidates_distance/spatial_step).astype(int)
                spatial_score = spatial_Weight[spatial_idx.reshape(-1)].view(np.shape(spatial_idx)[0], -1)                
                scores = rank_model.get_final_score(query_emb_tensor, poi_emb_tensor, spatial_score)
                top_indices = torch.topk(scores, min(100,poi_emb_tensor.size()[0]), largest=True)[1].cpu().numpy()
                top_indices = top_indices.tolist()
                for i in range(len(top_indices)):
                    predict_id = top_indices[i]
                    predict_poi_id = [poi_ids[j] for j in predict_id]
                    truth = np.array(qrels_test[test_Querys[batch_query_ids[i]].id])            
                    recall_100 = len(set(predict_poi_id[:100]) & set(truth)) / min(len(truth),100)
                    recall_50 = len(set(predict_poi_id[:50]) & set(truth)) / min(len(truth),50)
                    recall_20 = len(set(predict_poi_id[:20]) & set(truth)) / min(len(truth),20)
                    recall_10 = len(set(predict_poi_id[:10]) & set(truth)) / min(len(truth),10)
                    
                    recalls_100.append(recall_100)
                    recalls_50.append(recall_50)
                    recalls_20.append(recall_20)
                    recalls_10.append(recall_10)
                    partital_recalls_100.append(recall_100)
                    partital_recalls_50.append(recall_50)
                    partital_recalls_20.append(recall_20)
                    partital_recalls_10.append(recall_10)
                    
                    # label_onehot = torch.zeros(len(POIs)).to(args.device)
                    # label_onehot[truth] = 1
                    # totoal_poi_scores = torch.zeros(len(POIs)).to(args.device)
                    # totoal_poi_scores[predict_poi_id] = scores[i, predict_id]
                    
                    # ndcg_1 = retrieval_normalized_dcg(totoal_poi_scores, label_onehot, 1).item()
                    # ndcg_3 = retrieval_normalized_dcg(totoal_poi_scores, label_onehot, 3).item()
                    # ndcg_5 = retrieval_normalized_dcg(totoal_poi_scores, label_onehot, 5).item()
                    # ndcg_10 = retrieval_normalized_dcg(totoal_poi_scores, label_onehot, 10).item()
                    ndcg_1 = custom_ndcg(predict_poi_id,truth.tolist(), 1)
                    ndcg_3 = custom_ndcg(predict_poi_id,truth.tolist(), 3)
                    ndcg_5 = custom_ndcg(predict_poi_id,truth.tolist(), 5)
                    ndcg_10 = custom_ndcg(predict_poi_id,truth.tolist(), 10)                    
                    ndcgs_1.append(ndcg_1)
                    ndcgs_3.append(ndcg_3)
                    ndcgs_5.append(ndcg_5)
                    ndcgs_10.append(ndcg_10)
                    partital_ndcgs_1.append(ndcg_1)
                    partital_ndcgs_3.append(ndcg_3)
                    partital_ndcgs_5.append(ndcg_5)
                    partital_ndcgs_10.append(ndcg_10)                    
                    
            avg_ndcg_1 = np.mean(partital_ndcgs_1)
            avg_ndcg_3 = np.mean(partital_ndcgs_3)
            avg_ndcg_5 = np.mean(partital_ndcgs_5)
            avg_ndcg_10 = np.mean(partital_ndcgs_10)
                
            avg_recall_100 = np.mean(partital_recalls_100)
            avg_recall_50 = np.mean(partital_recalls_50)
            avg_recall_20 = np.mean(partital_recalls_20)
            avg_recall_10 = np.mean(partital_recalls_10)
                
            # 输出平均值
            logger.info(f"Average NDCG@1 in cluster {cluster_id}: {avg_ndcg_1}")
            logger.info(f"Average NDCG@3 in cluster {cluster_id}: {avg_ndcg_3}")
            logger.info(f"Average NDCG@5 in cluster {cluster_id}: {avg_ndcg_5}")
            logger.info(f"Average NDCG@10 in cluster {cluster_id}: {avg_ndcg_10}")
                
            logger.info(f"Average Recall@100 in cluster {cluster_id}: {avg_recall_100}")
            logger.info(f"Average Recall@50 in cluster {cluster_id}: {avg_recall_50}")
            logger.info(f"Average Recall@20 in cluster {cluster_id}: {avg_recall_20}")
            logger.info(f"Average Recall@10 in cluster {cluster_id}: {avg_recall_10}")
            
        # 重置列表

    avg_ndcg_1 = np.mean(ndcgs_1)
    avg_ndcg_3 = np.mean(ndcgs_3)
    avg_ndcg_5 = np.mean(ndcgs_5)
    avg_ndcg_10 = np.mean(ndcgs_10)
                
    avg_recall_100 = np.mean(recalls_100)
    avg_recall_50 = np.mean(recalls_50)
    avg_recall_20 = np.mean(recalls_20)
    avg_recall_10 = np.mean(recalls_10)       
     
    logger.info(f"recall@100: {avg_recall_100}")
    logger.info(f"recall@50: {avg_recall_50}")
    logger.info(f"recall@20: {avg_recall_20}")
    logger.info(f"recall@10: {avg_recall_10}")
    logger.info(f"NDCG@10: {avg_ndcg_10}")
    logger.info(f"NDCG@5: {avg_ndcg_5}")
    logger.info(f"NDCG@3: {avg_ndcg_3}")
    logger.info(f"NDCG@1: {avg_ndcg_1}")
    
    # cluster_count = 0    
    # file_path = 'path_to_your_file/'
    # cluster_count = 0
    # for cluster_id in range(len(test_query_to_cluster_inverted_id)):
    #     if len(test_query_to_cluster_inverted_id[cluster_id]) != 0 or poi_to_cluster_inverted_id[cluster_id] != 0:
    #         os.makedirs(os.path.join(args.origin_data_dir, 'clusters',str(cluster_count)), exist_ok=True)
    #         cluster_count = cluster_count + 1

    # cluster_count = 0
    # # for cluster_id in range(len(query_to_cluster_inverted_id)):
    # for cluster_id, query_ids in enumerate(test_query_to_cluster_inverted_id):
    #     if len(test_query_to_cluster_inverted_id[cluster_id]) != 0:
    #         with open(os.path.join(args.origin_data_dir, 'clusters', str(cluster_count), 'query_after_inference.csv'), 'w', newline='') as file:
    #             for query_id in tqdm(query_ids):
    #                 query = queries_after_inference[test_Querys[query_id].id]  # Assuming Querys is a dict where keys are query_ids
    #                 # if test_Querys[query_id].id == 16907:
    #                 #     print("16907 exist")
    #                 #     print(query['id'])                    
    #                 write_query_to_file(query, file)
    #         cluster_count = cluster_count +1
    # cluster_count = 0
    # for cluster_id, poi_ids in enumerate(poi_to_cluster_inverted_id):
    #     if len(poi_to_cluster_inverted_id[cluster_id]) != 0:
    #         with open(os.path.join(args.origin_data_dir, 'clusters', str(cluster_count), 'poi_after_inference.csv'), 'w', newline='') as file:
    #             for poi_id in tqdm(poi_ids):
    #                 poi = pois_after_inference[poi_id] 
    #                 write_poi_to_file(poi, file)    
    #         cluster_count = cluster_count + 1
    # exit(1)
    cluster_count = 0
    for cluster_id in range(len(test_query_to_cluster_inverted_id)):
        if len(test_query_to_cluster_inverted_id[cluster_id]) != 0 or len(poi_to_cluster_inverted_id[cluster_id]) != 0:
            os.makedirs(os.path.join(args.origin_data_dir, 'clusters',str(cluster_count)), exist_ok=True)
            cluster_count = cluster_count + 1

    cluster_count = 0
    # for cluster_id in range(len(query_to_cluster_inverted_id)):
    for cluster_id, query_ids in enumerate(test_query_to_cluster_inverted_id):
        if len(test_query_to_cluster_inverted_id[cluster_id]) != 0 or len(poi_to_cluster_inverted_id[cluster_id]) != 0:
            with open(os.path.join(args.origin_data_dir, 'clusters', str(cluster_count), 'query_after_inference.csv'), 'w', newline='') as file:
                for query_id in tqdm(query_ids):
                    query = queries_after_inference[test_Querys[query_id].id]  # Assuming Querys is a dict where keys are query_ids
                    # if test_Querys[query_id].id == 16907:
                    #     print("16907 exist")
                    #     print(query['id'])                    
                    write_query_to_file(query, file)
            cluster_count = cluster_count +1
    cluster_count = 0
    for cluster_id, poi_ids in enumerate(poi_to_cluster_inverted_id):
        if len(poi_to_cluster_inverted_id[cluster_id]) != 0 or len(test_query_to_cluster_inverted_id[cluster_id]) != 0:
            with open(os.path.join(args.origin_data_dir, 'clusters', str(cluster_count), 'poi_after_inference.csv'), 'w', newline='') as file:
                for poi_id in tqdm(poi_ids):
                    poi = pois_after_inference[poi_id] 
                    write_poi_to_file(poi, file)    
            cluster_count = cluster_count + 1
    
    print("cluster number: ", cluster_count)
if __name__ == "__main__":
    main()


