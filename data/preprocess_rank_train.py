import os
import json
import argparse
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import numpy as np
def load_rel(rel_path):
    reldict = defaultdict(list)
    rel_df = pd.read_csv(rel_path, sep=',')
    for index, row in rel_df.iterrows():
        qid, pid = row['query_id'], row['poi_id']
        reldict[qid].append(pid)
    return dict(reldict)

def preprocess(args):
    ir_tree_rank = json.load(open(args.data_dir + "bm25_{}_distance.json".format(args.alpha)))
    poi_dict = {}

    qrels_train = load_rel(args.data_dir + "qrels_train.csv")

    poi_path = args.data_dir + 'poi.csv'
    poi_df = pd.read_csv(poi_path, sep=',')

    for index, row in poi_df.iterrows():
        poi_id = int(row.id)
        poi_coor = eval(row.coor)
        poi_dict[poi_id] = [poi_coor, row.text]
    
    # 设定true和false的比例
    # ratio_true_false = 0.2

    # 定义向量的长度
    # np.random.seed(0)

    # vector_length = len(poi_dict)    
    # random_vector = np.random.choice([True, False], size=vector_length, p=[ratio_true_false, 1 - ratio_true_false]).tolist()


    print("poi loading over")
    count_line = 0
    jsonList = []

    query_train_df = pd.read_csv(args.data_dir + 'query_train.csv', sep=',', encoding='utf8')

    for index, row in tqdm(query_train_df.iterrows(), desc="Processing queries"):
        Item = dict()
        query_content = row['text']
        query_coordinate = eval(row['coor'])
        query_id = int(row['id'])
        Item['query_content'] = query_content
        Item['query_coor'] = query_coordinate
        Item['hard_negative_content'] = []

        for negative_id in ir_tree_rank[str(query_id)]:
            if negative_id not in qrels_train[query_id]:
                # if random_vector[negative_id]:
                [poi_coor, poi_text] = poi_dict[negative_id]
                Item['hard_negative_content'].append({"poi_coor": poi_coor, "poi_text": poi_text})

        Item['positive_content'] = []
        for positive_id in qrels_train[query_id]:
            # if random_vector[positive_id]:
            [poi_coor, poi_text] = poi_dict[positive_id]
            Item['positive_content'].append({"poi_coor": poi_coor, "poi_text": poi_text})

        if len(Item['positive_content'])!= 0 and len(Item['hard_negative_content'])!=0:
            jsonList.append(Item)
            count_line += 1

    print("process queries size is {}".format(count_line))
    jsonArr = json.dumps(jsonList, ensure_ascii=False)
    with open(args.output_data_dir + 'rank_train.json', 'w') as f2:
        f2.write(jsonArr)

def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_dir",
        default="./beijing/processed_data/",
        type=str,
        help="The input data directory",
    )

    parser.add_argument(
        "--alpha",
        default=0.4,
        type=float,
        help="The gpu device",
    )  
        
    parser.add_argument(
        "--output_data_dir",
        default="./beijing/processed_data/",
        type=str,
        help="The output data directory after preprocess",
    )

    args = parser.parse_args()

    return args

def main():
    args = get_arguments()
    preprocess(args)  



if __name__ == '__main__':
    main()


