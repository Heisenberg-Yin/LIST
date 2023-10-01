import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
import os
import pickle
import random
from random import sample
import csv
import pandas as pd
import numpy as np
import argparse
import utm
from tqdm import tqdm
random.seed(0)

def preprocess(args): 
    if not os.path.exists(args.output_data_dir):
        os.makedirs(args.output_data_dir)
        print(f"Create {args.output_data_dir}")
    else:
        print(f"{args.output_data_dir} exists")    
    
    poi_data_dir = args.data_dir + "attribute.csv"
    poi_map = {}
    poi_df = pd.read_csv(poi_data_dir,sep=',')    
    header = ['id', 'coor', 'text']
    poi_data = []
    
    for index, row in tqdm(poi_df.iterrows(), total=poi_df.shape[0], desc="reading poi"):        
        utm_ = utm.from_latlon(row.lat, row.lon)
        utm_x = utm_[0]
        utm_y = utm_[1]
        
        original_poi_id = int(row.poiId)
        
        if original_poi_id not in poi_map:
            poi_map[original_poi_id] = len(poi_data)
            poi_data.append([poi_map[original_poi_id],[utm_x,utm_y],row.text])
        else:
            print(poi_data[poi_map[original_poi_id]])
            print(row)
            exit(1)
    
    with open(args.output_data_dir + 'poi.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(poi_data)

    records_data_dir = args.data_dir + "records.csv"
    records_df = pd.read_csv(records_data_dir,sep=',')    
    query_map = {}
    query_id = 0
    header = ['id', 'coor', 'text']
    query_data = []
    records_header = ['query_id','poi_id']
    records_data = []    
    for index, row in tqdm(records_df.iterrows(), total=records_df.shape[0], desc="reading records"): 
        utm_ = utm.from_latlon(row.lat, row.lon)
        utm_x = utm_[0]
        utm_y = utm_[1]
        original_query_id = int(row.queryId)
        original_poi_id = int(row.poiId)
        if original_poi_id in poi_map:
            if original_query_id not in query_map:
                query_id = len(query_data)
                query_map[original_query_id] = []
                query_map[original_query_id].append([query_id, row.keywords, utm_x, utm_y])        
                query_data.append([query_id, [utm_x, utm_y], row.keywords])
            else:
                flag = True
                for item in query_map[original_query_id]:
                    if (row.keywords == item[1]) and (utm_x == item[2]) and (utm_y == item[3]):
                        flag = False
                        break
                if flag:
                    query_id = len(query_data)
                    query_map[original_query_id].append([query_id, row.keywords, utm_x, utm_y])        
                    query_data.append([query_id, [utm_x, utm_y], row.keywords])                

            for item in query_map[original_query_id]:
                if (row.keywords == item[1]) and (utm_x == item[2]) and (utm_y == item[3]):            
                    records_data.append([item[0], poi_map[original_poi_id]])
    
    with open(args.output_data_dir + 'query.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(query_data)
    
    with open(args.output_data_dir + 'qrels.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(records_header)
        writer.writerows(records_data)
    
    print("The query number is ", len(query_data))
    print("The poi number is ", len(poi_data))
    print("The records number is ", len(records_data))

def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_dir",
        default="./beijing/",
        type=str,
        help="The input data directory",
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