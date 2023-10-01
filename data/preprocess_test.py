# import os
# # import shutil
# import json
# import argparse
# # import subprocess
# # import multiprocessing
# # import numpy as np
# from tqdm import tqdm
# # import sys
# import pandas as pd
# import logging
# from collections import defaultdict
# # from transformers import BertTokenizer

# logger = logging.Logger(__name__, level=logging.INFO)
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s- %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

# def load_rel(rel_path):
#     reldict = defaultdict(list)
#     for line in tqdm(open(rel_path), desc=os.path.split(rel_path)[1]):
#         qid, pid = line.split('\t')
#         qid, pid = int(qid), int(pid)
#         reldict[qid].append((pid))
#     return dict(reldict)

# def preprocess(args):  
#     ir_tree_rank = json.load(open(args.data_dir+"bm25_distance.json"))
#     poi_dict = {}
#     qrels_test = load_rel(args.data_dir+"qrels_test.tsv")
#     poi_path = args.data_dir + 'poi.csv'
#     poi_df = pd.read_csv(poi_path,sep=',') 
#     for index, row in poi_df.iterrows():
#         poi_id = int(row.id)
#         poi_coor = eval(row.coor)
#         poi_dict[poi_id] = [poi_coor, row.text]
    
#     print("poi loading over")
#     count_line = 0
#     jsonList = []
    
#     with open(args.data_dir + 'query_test.tsv', 'r', encoding='utf8') as f:
#         for line in f:       
#             count_line = count_line + 1
#             if count_line % 1000 == 0:
#                 logger.info(count_line)
#             Item = dict()
#             line_arr = line.split('\t')
#             # print(line_arr)
#             # exit(1)
#             query_content = line_arr[2]
#             query_coordinate = eval(line_arr[1])
#             query_id = int(line_arr[0])
#             Item['query_content'] = query_content
#             Item['query_coor'] = query_coordinate
#             Item['hard_negative_content'] = []
#             for negative_id in ir_tree_rank[str(query_id)]:
#                 if negative_id not in qrels_test[query_id]:
#                     [poi_coor, poi_text] = poi_dict[negative_id]                    
#                     Item['hard_negative_content'].append({"poi_coor": poi_coor, "poi_text": poi_text})
#             Item['positive_content'] = []
#             for positive_id in qrels_test[query_id]:
#                 [poi_coor, poi_text] = poi_dict[positive_id] 
#                 Item['positive_content'].append({"poi_coor": poi_coor, "poi_text": poi_text})
#             jsonList.append(Item)
    
#     jsonArr = json.dumps(jsonList, ensure_ascii=False)
#     f2 = open(args.out_data_dir + 'val.json', 'w')
#     f2.write(jsonArr)
#     f2.close()
    
# def get_arguments():
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument(
#         "--data_dir",
#         default="./beijing/processed_data/",
#         type=str,
#         help="The input data directory",
#     )
    
#     parser.add_argument(
#         "--out_data_dir",
#         default="./beijing/processed_data/",
#         type=str,
#         help="The output data directory after preprocess",
#     )

#     args = parser.parse_args()

#     return args


# def main():
#     args = get_arguments()
#     if not os.path.exists(args.out_data_dir):
#         os.makedirs(args.out_data_dir)
#     preprocess(args)  



# if __name__ == '__main__':
#     main()