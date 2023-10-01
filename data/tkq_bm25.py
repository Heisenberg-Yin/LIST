'''
    This is a simple baseline that test the performance of bm25.
    It compute the query score to all the POIs and save the scores as a sparse matrix.
    The next time of calling we just load the matrix and do the ranking.
'''
import argparse
import os
import csv
import jieba
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics.functional import retrieval_normalized_dcg
from rank_bm25 import BM25Okapi
import json
bm25 = None

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

def bm25_search(query):
    tokenized_query = list(jieba.cut(query))
    doc_scores = bm25.get_scores(tokenized_query)
    return doc_scores

def preprocess(args): 
    global bm25
    poi_path = args.data_dir + 'poi.csv'

    query_path = args.data_dir + 'query.csv'

    test_query_path = args.data_dir + 'query_test.csv'
    ground_truth_path = args.data_dir + 'qrels.csv'

    min_x = 1e9
    max_x = -1e9
    min_y = 1e9
    max_y = -1e9

    poi_locations = []
    poi_txt = []

    # normalize the embeddings
    count = 0
    poi_df = pd.read_csv(poi_path,sep=',') 
    for index, row in poi_df.iterrows():
        count = count+1
        xy_tuple = eval(row.coor)
        x = xy_tuple[0]
        y = xy_tuple[1]
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        poi_locations.append((x, y))        
        current = []
        current.extend(jieba.cut(row.text))     
        poi_txt.append(current)
    bm25 = BM25Okapi(poi_txt)
    poi_locations = np.array(poi_locations)

    print("max_x:",max_x)
    print("min_x:",min_x)
    print("max_y:",max_y)
    print("min_y:",min_y)


    query_locations = []
    query_txt = []
    # normalize the embeddings
    query_df = pd.read_csv(query_path,sep=',') 
    for index, row in query_df.iterrows():
        xy_tuple = eval(row.coor)
        x = xy_tuple[0]
        y = xy_tuple[1]
        query_locations.append((x, y))
        query_txt.append((row.text))



    query_locations = np.array(query_locations)

    d_norm = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

    #print("d_norm is %f", d_norm) 227689.23857405974
    print("d_norm is ", d_norm)


    query_result = {}
    duplicate_truth = 0
    
    records_df = pd.read_csv(ground_truth_path,sep=',') 
    for index, row in records_df.iterrows():
        query_id = int(row.query_id)
        result_id = int(row.poi_id)
        if query_id not in query_result:
            query_result[query_id] = []
        query_result[query_id].append(result_id)
        if len(query_result[query_id]) > 1:
            duplicate_truth += 1
        # if query_id not in query_map:
            # query_map[query_id] = len(query_map)
            
    print('total query', len(query_result))
    print('duplicate_truth', duplicate_truth)

    bm25_path = args.data_dir + './bm25_scores.dat'.format(args.alpha)
    
    max_bm25_score = 0
    min_bm25_score = 1e9
    
    if os.path.exists(bm25_path):
        print('Loading bm25 scores...')
        bm25_scores = np.memmap(bm25_path, dtype='float16', mode='r', shape=(len(query_txt), len(poi_txt)))
        # if the bm25 scores are already computed, we can just load it
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        bm25_scores = np.memmap(bm25_path, dtype='float16', mode='w+', shape=(len(query_txt), len(poi_txt)))
        BATCH_SIZE = 1000
        with ProcessPoolExecutor(max_workers=16) as executor:
            for i in range(0, len(query_txt), BATCH_SIZE):
                batch_queries = query_txt[i:i + BATCH_SIZE]
                futures_to_index = {executor.submit(bm25_search, query): idx for idx, query in enumerate(batch_queries)}
                for future in tqdm(as_completed(futures_to_index), total=len(batch_queries), desc=f"处理查询 {i} 到 {i+BATCH_SIZE}"):
                    scores = future.result()
                    index = i + futures_to_index[future]
                    bm25_scores[index, :] = scores
        bm25_scores.flush()
            # queries = [None] * len(query_map)

            # for query in query_result:
            #     queries[query_map[query]] = query_txt[query]
            #     # print(query_txt[query])
            # futures_to_index = {executor.submit(bm25_search, query): idx for idx, query in enumerate(queries)}

            # # Display progress bar and collect results
            # for future in tqdm(as_completed(futures_to_index), total=len(queries), desc="Processing queries"):
            #     scores = future.result()
            #     index = futures_to_index[future]
            #     bm25_scores[index, :] = scores
            #     bm25_scores.flush()              
                
                
    max_bm25_score = bm25_scores.max(axis=None)
    min_bm25_score = bm25_scores.min(axis=None)        
    print("max bm25 score is {}, min bm25 score is {}".format(max_bm25_score, min_bm25_score))
    query_locations = torch.from_numpy(query_locations).to(args.device)
    poi_locations = torch.from_numpy(poi_locations).to(args.device)

    recalls_100 = []
    recalls_50 = []
    recalls_20 = []
    recalls_10 = []

    ndcgs_1 = []
    ndcgs_3 = []
    ndcgs_5 = []
    ndcgs_10 = []

    json_saved = {}
    
    
    if not os.path.exists(args.output_data_dir + "bm25_{}_distance.json".format(args.alpha)):
        # json.dump(json_saved, json_file)        
        # Now for each query, we search for the top-k nearest POIs and rerank them
        for query in tqdm(query_result):
            # The predict score is 𝑆𝑇(𝑝,𝑞) =(1−𝛼)×(1−𝑆𝐷𝑖𝑠𝑡(𝑝.𝑙𝑜𝑐,𝑞.𝑙𝑜𝑐))+ 𝛼 ×𝑇𝑅𝑒𝑙(𝑝.𝑑𝑜𝑐,𝑞.𝑑𝑜𝑐)
            # where 𝑆𝐷𝑖𝑠𝑡(𝑝.𝑙𝑜𝑐,𝑞.𝑙𝑜𝑐) is the distance similarity between the location of POI p and the location of query q
            # and 𝑇𝑅𝑒𝑙(𝑝.𝑑𝑜𝑐,𝑞.𝑑𝑜𝑐) is the bm25 score between the description of POI p and the description of query q.
            # and 𝛼 is a hyper-parameter to balance the two parts (we set 𝛼 = 0.4 in our experiments).
            
            query_loc = query_locations[query]
            # First compute the distance similarity
            dist_sim = torch.sum((poi_locations - query_loc) ** 2, dim=1).sqrt()
            dist_sim = 1 - dist_sim / d_norm
            # Then compute the description similarity
            desc_sim = bm25_scores[query, :].copy()
            desc_sim = torch.from_numpy(desc_sim).to(args.device)

            predict_score = (1 - args.alpha) * dist_sim + args.alpha * (desc_sim - min_bm25_score)/ (max_bm25_score - min_bm25_score)
            # predict_score = predict_score.cpu()
            top_indices = torch.topk(predict_score, 100, largest=True)[1].cpu().numpy()
            json_saved[query] = top_indices.tolist()

            truth = query_result[query]

            truth = np.array(truth)
            
            recall_100 = len(set(top_indices[:100]) & set(truth)) / len(truth)
            recall_50 = len(set(top_indices[:50]) & set(truth)) / len(truth)
            recall_20 = len(set(top_indices[:20]) & set(truth)) / len(truth)
            recall_10 = len(set(top_indices[:10]) & set(truth)) / len(truth)

            recalls_100.append(recall_100)
            recalls_50.append(recall_50)
            recalls_20.append(recall_20)
            recalls_10.append(recall_10)
            
            # onehot_truth = query_result_matrix[query_map[query]].to(args.device)
            ndcg_1 = custom_ndcg(top_indices,truth.tolist(), 1)
            ndcg_3 = custom_ndcg(top_indices,truth.tolist(), 3)
            ndcg_5 = custom_ndcg(top_indices,truth.tolist(), 5)
            ndcg_10 = custom_ndcg(top_indices,truth.tolist(), 10)            
            # ndcg_1 = retrieval_normalized_dcg(predict_score, onehot_truth, 1).item()
            # ndcg_3 = retrieval_normalized_dcg(predict_score, onehot_truth, 3).item()
            # ndcg_5 = retrieval_normalized_dcg(predict_score, onehot_truth, 5).item()
            # ndcg_10 = retrieval_normalized_dcg(predict_score, onehot_truth, 10).item()
            
            ndcgs_1.append(ndcg_1)
            ndcgs_3.append(ndcg_3)
            ndcgs_5.append(ndcg_5)
            ndcgs_10.append(ndcg_10)

        with open(args.output_data_dir + "bm25_{}_distance.json".format(args.alpha), "w") as json_file:
            json.dump(json_saved, json_file)


        print('recall@100', np.mean(recalls_100))
        print('recall@50', np.mean(recalls_50))
        print('recall@20', np.mean(recalls_20))
        print('recall@10', np.mean(recalls_10))

        print('ndcg@1', np.mean(ndcgs_1))
        print('ndcg@3', np.mean(ndcgs_3))
        print('ndcg@5', np.mean(ndcgs_5))
        print('ndcg@10', np.mean(ndcgs_10))
    # max_bm25_score = bm25_scores.max(axis=None)
    # min_bm25_score = bm25_scores.min(axis=None)
    recalls_100_test = []
    recalls_50_test = []
    recalls_20_test = []
    recalls_10_test = []

    ndcgs_1_test = []
    ndcgs_3_test = []
    ndcgs_5_test = []
    ndcgs_10_test = []
    # test_query_locations = []
    # test_query_txt = []     
    # test_query_df = pd.read_csv(test_query_path,sep=',') 
    test_ground_truth_path = args.data_dir + 'qrels_test.csv'
    # for index, row in test_query_df.iterrows():
        # xy_tuple = eval(row.coor)
        # x = xy_tuple[0]
        # y = xy_tuple[1]
        # test_query_locations.append((x, y))
        # test_query_txt.append((row.text))
    # test_query_locations = np.array(test_query_locations)
    # test_query_locations = torch.from_numpy(test_query_locations).to(args.device)

    test_query_result = {}
    test_duplicate_truth = 0
    
    test_records_df = pd.read_csv(test_ground_truth_path,sep=',') 
    for index, row in test_records_df.iterrows():
        query_id = int(row.query_id)
        result_id = int(row.poi_id)
        if query_id not in test_query_result:
            test_query_result[query_id] = []
        test_query_result[query_id].append(result_id)
        if len(test_query_result[query_id]) > 1:
            test_duplicate_truth += 1
        # if query_id not in test_query_map:
            # test_query_map[query_id] = len(test_query_map)
            
    print('total query', len(test_query_result))
    print('duplicate_truth', test_duplicate_truth)

    for test_query in tqdm(test_query_result):
        # The same formula is applied to compute the score
        test_query_loc = query_locations[test_query]
        dist_sim_test = torch.sum((poi_locations - test_query_loc) ** 2, dim=1).sqrt()
        dist_sim_test = 1 - dist_sim_test / d_norm
        desc_sim_test = bm25_scores[test_query, :].copy()
        desc_sim_test = torch.from_numpy(desc_sim_test).to(args.device)
        
        predict_score_test = (1 - args.alpha) * dist_sim_test + args.alpha * (desc_sim_test - min_bm25_score)/ (max_bm25_score - min_bm25_score)
        top_indices_test = torch.topk(predict_score_test, 100, largest=True)[1].cpu().numpy()

        truth_test = test_query_result[test_query]
        truth_test = np.array(truth_test)
        
        recall_100_test = len(set(top_indices_test[:100]) & set(truth_test)) / len(truth_test)
        recall_50_test = len(set(top_indices_test[:50]) & set(truth_test)) / len(truth_test)
        recall_20_test = len(set(top_indices_test[:20]) & set(truth_test)) / len(truth_test)
        recall_10_test = len(set(top_indices_test[:10]) & set(truth_test)) / len(truth_test)

        recalls_100_test.append(recall_100_test)
        recalls_50_test.append(recall_50_test)
        recalls_20_test.append(recall_20_test)
        recalls_10_test.append(recall_10_test)
        
        ndcg_1_test = custom_ndcg(top_indices_test,truth_test.tolist(), 1)
        ndcg_3_test = custom_ndcg(top_indices_test,truth_test.tolist(), 3)
        ndcg_5_test = custom_ndcg(top_indices_test,truth_test.tolist(), 5)
        ndcg_10_test = custom_ndcg(top_indices_test,truth_test.tolist(), 10) 
        
        ndcgs_1_test.append(ndcg_1_test)
        ndcgs_3_test.append(ndcg_3_test)
        ndcgs_5_test.append(ndcg_5_test)
        ndcgs_10_test.append(ndcg_10_test)

    print('Test Metrics:')
    print('recall@100', np.mean(recalls_100_test))
    print('recall@50', np.mean(recalls_50_test))
    print('recall@20', np.mean(recalls_20_test))
    print('recall@10', np.mean(recalls_10_test))

    print('ndcg@1', np.mean(ndcgs_1_test))
    print('ndcg@3', np.mean(ndcgs_3_test))
    print('ndcg@5', np.mean(ndcgs_5_test))
    print('ndcg@10', np.mean(ndcgs_10_test))

def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_dir",
        default="./beijing/processed_data/",
        type=str,
        help="The input data directory",
    )
    
    parser.add_argument(
        "--output_data_dir",
        default="./beijing/processed_data/",
        type=str,
        help="The output data directory after preprocess",
    )
    
    parser.add_argument(
        "--device",
        default=0,
        type=int,
        help="The gpu device",
    )
    parser.add_argument(
        "--alpha",
        default=0.4,
        type=float,
        help="The gpu device",
    )  
        
    args = parser.parse_args()

    return args




def main():
    args = get_arguments()
    torch.cuda.set_device(args.device)
    preprocess(args)  
    

if __name__ == '__main__':
    main()



