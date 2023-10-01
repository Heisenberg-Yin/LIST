import torch
from src.rank.model.LISTR import LISTR
import logging
# from src.inference_dual_encoder_args import run_parse_args
from src.utils import load_states_from_checkpoint, set_seed, is_first_worker
import numpy as np
import os
from transformers import BertTokenizer
import torch.nn.functional as F
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import time
from typing import Tuple, List
# from src.dataset.dataset import EvalDataset
from torch.utils.data import DataLoader, SequentialSampler, Dataset
from tqdm import tqdm
import pathlib
import pickle
import argparse
import collections
import csv
logger = logging.getLogger(__name__)

Embedding_BiEncoderPoi = collections.namedtuple("BiEncoderPoi", ["id", "coor", "text"])

def pack_tensor_2D(lstlst, default, dtype, length=None):
    batch_size = len(lstlst)
    length = length if length is not None else max(len(l) for l in lstlst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, l in enumerate(lstlst):
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
    return tensor


class Query():
    def __init__(self, id, embedding: np.array, coordinate: List[object], attention: List[object]):
        self.id = id
        self.embedding = embedding
        self.coordinate = coordinate
        self.attention = attention

class Poi():
    def __init__(self, id, embedding: np.array, coordinate: List[object]):
        self.id = id
        self.embedding = embedding
        self.coordinate = coordinate
        # self.attention = attention
        
        
class EvalDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length = 96):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data()        
        
    def load_data(self):
        
        df = pd.read_csv(self.file_path, sep=',')
        def create_object(id: int, coor: List, text: str):
            return Embedding_BiEncoderPoi(
                id, 
                coor,    
                text
            )
        objects = []
        for index, row in df.iterrows():
            id = int(row.id)
            text = row.text
            coor = eval(row.coor)
            objects.append(create_object(id = id, coor = coor, text = text))
            
        logger.info(
            "Total data size: {} ".format(len(objects))
        )
        return objects

    def __getitem__(self, index):
        object = self.data[index]

        sentence_encode = self.tokenizer.encode_plus(object.text, max_length=self.max_length, truncation=True , padding=False, 
                                                           return_attention_mask=True, return_token_type_ids=True)
        coor = object.coor
        id = object.id
        
        input_ids = sentence_encode['input_ids']
        attn_mask = sentence_encode['attention_mask']
        token_type_ids = sentence_encode['token_type_ids']
        
        return id, input_ids, attn_mask, token_type_ids, coor
    
    def __len__(self):
        return len(self.data)
    
    def get_collate_fn(self):
        def create_biencoder_input2(features):
            id_list = []            
            input_ids_list = []
            attn_mask_list = []            
            token_type_ids_list = []
            coor_list = []

            for index, feature in enumerate(features):
                id_list.append(feature[0])               
                input_ids_list.append(feature[1])
                attn_mask_list.append(feature[2])
                token_type_ids_list.append(feature[3])                
                coor_list.append(feature[4])

            coordinates = np.concatenate([np.array(coor).reshape(1, -1) for coor in coor_list], axis=0)
            
            input_ids_tensor = pack_tensor_2D(input_ids_list, default=0, 
                dtype=torch.int64, length=self.max_length)
            
            attn_mask_tensor = pack_tensor_2D(attn_mask_list, default=0, 
                dtype=torch.int64, length=self.max_length)            
            
            token_types_tensor = pack_tensor_2D(token_type_ids_list, default=0, 
                dtype=torch.int64, length=self.max_length)
            
            return {
                    'embedding': [
                                  id_list, input_ids_tensor, attn_mask_tensor, token_types_tensor, coordinates],
                    }            
        return create_biencoder_input2


def run_parse_args():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--log_dir", type=str)
    # parser.add_argument("--embedding_file", type=str)
    parser.add_argument("--checkpoint_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--model_type", default=None, type=str, help="config name for model initialization")
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_poi_length", type=int, default=96)
    parser.add_argument("--dataset", default="beijing", type=str, help="dataset to be used")    
    parser.add_argument('--n_heads', type=int, default=3)
    parser.add_argument('--att_dropout', type=float, default=0)    
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
            "--origin_data_dir",
            default=None,
            type=str,
    )
    parser.add_argument('--share_weight', action='store_true') 
    parser.add_argument('--bert_dropout', type=float, default=0)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument('--spatial_dropout', type=float, default=0)
    parser.add_argument('--spatial_step_k', type=int, default=1000)

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
        if not os.path.exists(os.path.join(args.log_dir,args.dataset)):
            os.makedirs(os.path.join(args.log_dir,args.dataset))
     
    args.log_dir = os.path.join(args.log_dir,args.dataset)
    return args

def load_model(args):

    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  

    
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
    
def generate_dense_vectors(args, model, tokenizer) -> List[Tuple[object, np.array]]:
    args.batch_size = args.per_gpu_batch_size*max(1, args.n_gpu)
    logger.info("***** Generating Dense Vectors *****")
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.batch_size
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    poi_dataset = EvalDataset(args.origin_data_dir + "poi.csv", tokenizer,
                                    max_length=args.max_poi_length)
    poi_sampler = SequentialSampler(poi_dataset)
    poi_dataloader = DataLoader(poi_dataset, sampler=poi_sampler,
                        collate_fn=poi_dataset.get_collate_fn(),
                        batch_size=args.batch_size)
    
    T1 = time.perf_counter()
    total = 0
    results = []
    POIs = []
    for batch_id, batch in enumerate(tqdm(poi_dataloader)):  
        model.eval()
        batch_embeddings = batch['embedding']     
        poi_id_list = batch_embeddings[0]
        inputs_retriever = {
            "poi_ids": batch_embeddings[1].long().to(args.device),
            "poi_attn_mask": batch_embeddings[2].long().to(args.device),
            "poi_segments": batch_embeddings[3].long().to(args.device),
        }          
        poi_coordinates = batch_embeddings[4]
        
        with torch.no_grad():
            out = model.get_poi_representation(**inputs_retriever)
                        
        out = out.float().cpu()
        
        poi_ids = poi_id_list

        assert len(poi_ids) == out.size(0)

        total += len(poi_ids)

        for i in range(out.size(0)):
            results.extend([(poi_ids[i], out[i].view(-1).numpy())])
            POIs.append(Poi(id=poi_ids[i], embedding=out[i].view(-1).numpy(), coordinate=poi_coordinates[i,:].tolist()))
    print(total)
    T2 = time.perf_counter()
    print("Program execution time: %s milliseconds" % ((T2 - T1)*1000))
    file = os.path.join(args.output_dir, 'poi_embedding.pkl')
    
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info('Writing results to %s' % file)
    with open(file, mode='wb') as f:
        pickle.dump(results, f)
    
    
    with open(args.output_dir + 'poi_after_inference.csv', 'w', newline='') as file:
        for poi in tqdm(POIs):
            docstring = "{},{},{},".format(poi.id,poi.coordinate[0],poi.coordinate[1])
            poi_emb= poi.embedding
            for j in range(len(poi_emb)):
                docstring = docstring+str(poi_emb[j])
                if j != len(poi_emb) - 1 :
                    docstring = docstring + " "
            docstring = docstring + "\n"            
            file.write(docstring)    
    
    
    query_dataset = EvalDataset(args.origin_data_dir + "query.csv", tokenizer,
                                    max_length=args.max_query_length)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler,
                        collate_fn=query_dataset.get_collate_fn(),
                        batch_size=args.batch_size)
    
    T1 = time.perf_counter()
    total = 0
    results = []
    Querys = []
    for batch_id, batch in enumerate(tqdm(query_dataloader)):  
        model.eval()
        batch_embeddings = batch['embedding']     
        query_id_list = batch_embeddings[0]
        inputs_retriever = {
            "query_ids": batch_embeddings[1].long().to(args.device),
            "query_attn_mask": batch_embeddings[2].long().to(args.device),
            "query_segments": batch_embeddings[3].long().to(args.device),
        }          
        query_coordinates = batch_embeddings[4]
        T1 = time.perf_counter()
        with torch.no_grad():
            out = model.get_query_representation(**inputs_retriever)
            attention = model.get_attention_score(out)
        T2 = time.perf_counter()
        # print("Program execution time: %s milliseconds" % ((T2 - T1)*1000))
        out = out.float().cpu()
        attention = attention.float().cpu()
        query_ids = query_id_list

        assert len(query_ids) == out.size(0)

        total += len(query_ids)

        for i in range(out.size(0)):
            Querys.append(Query(id=query_ids[i], embedding=out[i].view(-1).numpy(), coordinate=query_coordinates[i,:].tolist(), attention=attention[i,:].tolist()))
            results.extend([(query_ids[i], out[i].view(-1).numpy(), attention[i].view(-1).numpy())])

    print(total)
    T2 = time.perf_counter()
    print("Program execution time: %s milliseconds" % ((T2 - T1)*1000))
    file = os.path.join(args.output_dir, 'query_embedding.pkl')     
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info('Writing results to %s' % file)
    with open(file, mode='wb') as f:
        pickle.dump(results, f)
        
    
    with open(args.output_dir + 'query_after_inference.csv', 'w', newline='') as file:
        for query in tqdm(Querys):
            docstring = "{},{},{},{},{},".format(query.id,query.coordinate[0],query.coordinate[1],query.attention[0],query.attention[1])
            query_emb= query.embedding
            for j in range(len(query_emb)):
                docstring = docstring+str(query_emb[j])
                if j != len(query_emb) - 1 :
                    docstring = docstring + " "
            docstring = docstring + "\n"            
            file.write(docstring)
    
 
    return 

def main():
    args = run_parse_args() 
    set_env(args)    
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
    
    spatial_weight = F.relu(model.spatial_model.weight).tolist()
    spatial_weight = [weight[0] for weight in spatial_weight]
    spatial_arr = [0]
    for weight in spatial_weight:
        spatial_arr.append(spatial_arr[-1] + weight)
    with open(args.output_dir + 'spatial_array.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(spatial_arr)   

    generate_dense_vectors(args, model, tokenizer)   


if __name__ == "__main__":
    main()


