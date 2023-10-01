# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O
# import os
# import pickle
# import random
# from random import sample
# import csv
# import pandas as pd
# import numpy as np
# import argparse
# import utm
# from tqdm import tqdm
# import json
# from typing import List
# # from collections import defaultdict
# random.seed(0)

# class geo_object():
#     def __init__(self, id: int, text: str, coordinate: List[object]):
#         self.id = id
#         self.text = text
#         self.coordinate = coordinate
        
# def preprocess(args): 
#     if not os.path.exists(args.output_data_dir):
#         os.makedirs(args.output_data_dir)
#         print(f"Create {args.output_data_dir}")
#     else:
#         print(f"{args.output_data_dir} exists")    
    
    
#     # poi_data_file = os.path.join(args.data_dir, "docs.json")
#     # poi_data = []
#     # poi_count = 0
#     # with open(poi_data_file, 'r', encoding='utf-8') as file:
#     #     for line in tqdm(file.readlines()):
#     #         dic = json.loads(line)
#     #         poi_id = dic['doc_id']
#     #         text = dic['address']
#     #         coor = eval(dic['gis'])[-1]
#     #         if coor != "":
#     #             coor = list(eval(coor))
#     #         # print(coor)
#     #         utm_ = utm.from_latlon(coor[1], coor[0])
#     #         utm_x = utm_[0]
#     #         utm_y = utm_[1] 
#     #         poi_data.append([poi_id,[utm_x,utm_y],text])
#     #         # pois.append(geo_object(poi_id, text, [utm_x, utm_y]))
#     #         assert poi_id == poi_count
#     #         poi_count = poi_count + 1
    
#     header = ['id', 'coor', 'text']
    
#     # with open(os.path.join(args.output_data_dir, 'poi.csv'), 'w', newline='') as csvfile:
#     #     writer = csv.writer(csvfile)
#     #     writer.writerow(header)
#     #     writer.writerows(poi_data)

#     train_data_file =os.path.join(args.data_dir, "train.json")
#     train_querys_data = []
#     train_count = 0
#     train_qrels = []
#     querys = []
#     with open(train_data_file, 'r', encoding='utf-8') as file:
#         for line in file.readlines():
#             dic = json.loads(line)
#             text= dic['query']
#             coor = eval(dic['query_gis'])[-1]
#             if coor != "":
#                 coor = list(eval(coor))
#             else:
#                 continue
#             utm_ = utm.from_latlon(coor[1], coor[0])
#             utm_x = utm_[0]
#             utm_y = utm_[1] 
#             train_querys_data.append([train_count,[utm_x,utm_y],text])
#             # querys.append([train_count,[utm_x,utm_y],text])
#             # print(coor)            
#             # train_querys.append(geo_object(train_count, text, coor))
            
#             if isinstance(dic['pos_id'], int):
#                 # train_positive_pair[train_count].append(dic['pos_id'])
#                 train_qrels.append([train_count, dic['pos_id']])
#             else:
#                 print(eval(dic['pos_id'])) 
#                 print("error")
#                 exit(1)
#             train_count = train_count + 1

#     print("Train Querys number is {}".format(train_count))
#     print("Train Qrels number is {}".format(len(train_qrels)))
#     with open(os.path.join(args.output_data_dir, 'query_train.csv'), 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(header)
#         writer.writerows(train_querys_data)
    
#     qrels_header = ['query_id', 'poi_id']
#     qrels = []
#     with open(os.path.join(args.output_data_dir, 'qrels_train.csv'), 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(qrels_header)
#         writer.writerows(train_qrels)
                                
#     qrels.extend(train_qrels)
#     querys.extend(train_querys_data)
#     val_data_file = os.path.join(args.data_dir, "dev.json")
#     val_querys_data = []
#     val_count = train_count
#     # val_positive_pair = defaultdict(list)
#     val_qrels = []
#     with open(val_data_file, 'r', encoding='utf-8') as file:
#         for line in file.readlines():
#             dic = json.loads(line)
#             text= dic['query']
#             coor = eval(dic['query_gis'])[-1]
#             if coor != "":
#                 coor = list(eval(coor))
#             else:
#                 continue
#             # print(coor)            
#             # val_querys_data.append(geo_object(val_count, text, coor))
#             utm_ = utm.from_latlon(coor[1], coor[0])
#             utm_x = utm_[0]
#             utm_y = utm_[1] 
#             val_querys_data.append([val_count,[utm_x,utm_y],text])
#             # querys.append([val_count,[utm_x,utm_y],text])
#             if isinstance(dic['pos_id'], int):
#                 # val_positive_pair[val_count].append(dic['pos_id'])
#                 val_qrels.append([val_count, dic['pos_id']])
#             else:
#                 print(eval(dic['pos_id'])) 
#                 print("error")
#                 exit(1)
#             val_count = val_count + 1
#     qrels.extend(val_qrels)
#     querys.extend(val_querys_data)
                
#     combined = list(zip(val_querys_data, val_qrels))
#     random.shuffle(combined)
#     val_querys_data[:], val_qrels[:] = zip(*combined)
#     split_index = len(val_querys_data) // 2
#     test_querys_data, test_qrels = val_querys_data[split_index:], val_qrels[split_index:]
#     val_querys_data, val_qrels = val_querys_data[:split_index], val_qrels[:split_index]
    

#     print("Val Querys number is {}".format(len(val_querys_data)))
#     print("Val Qrels number is {}".format(len(val_qrels)))
#     print("Test Querys number is {}".format(len(test_querys_data)))
#     print("Test Qrels number is {}".format(len(test_qrels)))
        
#     with open(os.path.join(args.output_data_dir, 'query_val.csv'), 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(header)
#         writer.writerows(val_querys_data)
    
#     with open(os.path.join(args.output_data_dir, 'qrels_val.csv'), 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(qrels_header)
#         writer.writerows(val_qrels)
    
#     with open(os.path.join(args.output_data_dir, 'query_test.csv'), 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(header)
#         writer.writerows(test_querys_data)
    
#     with open(os.path.join(args.output_data_dir, 'qrels_test.csv'), 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(qrels_header)
#         writer.writerows(test_qrels)
#     # test_data_file = os.path.join(args.data_dir, "test.json")
#     # test_querys_data = []
#     # test_count = val_count
#     # test_positive_pair = defaultdict(list)
#     # test_qrels = []
#     # with open(test_data_file, 'r', encoding='utf-8') as file:
#     #     for line in file.readlines():
#     #         dic = json.loads(line)
#     #         text= dic['query']
#     #         coor = eval(dic['query_gis'])[-1]
#     #         if coor != "":
#     #             coor = list(eval(coor))
#     #         else:
#     #             continue                
#     #         # print(coor)            
#     #         # test_querys.append(geo_object(test_count, text, coor))
#     #         utm_ = utm.from_latlon(coor[1], coor[0])
#     #         utm_x = utm_[0]
#     #         utm_y = utm_[1] 
#     #         test_querys_data.append([test_count,[utm_x,utm_y],text])
#     #         # querys.append([test_count,[utm_x,utm_y],text])
#     #         # print(dic['idx'])
#     #         if "pos_id" in dic:
#     #         # print(dic)
#     #             if isinstance(dic['pos_id'], int):
#     #                 test_qrels.append([test_count, dic['pos_id']])
#     #             else:
#     #                 print(eval(dic['pos_id'])) 
#     #                 print("error")
#     #                 exit(1)
#     #         else:
#     #             continue
#     #         test_count = test_count + 1    
#     # qrels.extend(test_qrels)
#     # print("Test Querys number is {}".format(test_count - val_count))
#     # print("Test Qrels number is {}".format(len(test_qrels)))
#     print("Querys number is {}".format(len(querys)))
#     print("Qrels number is {}".format(len(qrels)))
    
#     # with open(os.path.join(args.output_data_dir, 'query_test.csv'), 'w', newline='') as csvfile:
#     #     writer = csv.writer(csvfile)
#     #     writer.writerow(header)
#     #     writer.writerows(test_querys_data)

#     with open(os.path.join(args.output_data_dir, 'query.csv'), 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(header)
#         writer.writerows(querys)
        
#     # with open(os.path.join(args.output_data_dir, 'qrels_test.csv'), 'w', newline='') as csvfile:
#     #     writer = csv.writer(csvfile)
#     #     writer.writerow(qrels_header)
#     #     writer.writerows(test_qrels)

#     with open(os.path.join(args.output_data_dir, 'qrels.csv'), 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(qrels_header)
#         writer.writerows(qrels)
          

# def get_arguments():
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument(
#         "--data_dir",
#         default="./data/geo-glue/",
#         type=str,
#         help="The input data directory",
#     )
    
#     parser.add_argument(
#         "--output_data_dir",
#         default="./data/geo-glue/processed_data/",
#         type=str,
#         help="The output data directory after preprocess",
#     )

#     args = parser.parse_args()

#     return args

# def main():
#     args = get_arguments()
#     preprocess(args)  
    

# if __name__ == '__main__':
#     main()