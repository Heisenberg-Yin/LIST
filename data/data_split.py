import pandas as pd # data processing, CSV file I/O
import random
from random import sample
import csv
import argparse
random.seed(0)

def preprocess(args):         

    query_data_dir = args.data_dir + "query.csv"
    query_df = pd.read_csv(query_data_dir,sep=',')
    
    qrels_data_dir = args.data_dir + "qrels.csv"
    
    qrels_df = pd.read_csv(qrels_data_dir,sep=',')
    unique_query_id_set = set()
    for index, row in qrels_df.iterrows():
        unique_query_id_set.add(int(row['query_id']))   
    
    unique_query_id_list = list(unique_query_id_set)
    print("len of unique query if of all qrels: {}".format(len(unique_query_id_list)))

    unique_query_id_num = len(unique_query_id_list)

    unique_query_id_candidates = range(unique_query_id_num)

    test_query_id_list = list(sample(unique_query_id_candidates, int(0.1*len(unique_query_id_candidates))))

    train_val_id_candidates = list(set(unique_query_id_candidates).difference(test_query_id_list))

    val_query_id_list = list(sample(train_val_id_candidates, int(0.1*len(train_val_id_candidates))))
    
    train_query_id_list = list(set(train_val_id_candidates).difference(val_query_id_list))
    
    train_qrels_query_id = [unique_query_id_list[idx] for idx in train_query_id_list]
    val_qrels_query_id = [unique_query_id_list[idx] for idx in val_query_id_list]
    test_qrels_query_id = [unique_query_id_list[idx] for idx in test_query_id_list]  
    
    query_train_id = set(train_qrels_query_id)
    query_val_id = set(val_qrels_query_id)
    query_test_id = set(test_qrels_query_id)
    train_qrels_account = val_qrels_account = test_qrels_account = 0

    with open(args.output_data_dir + 'qrels_train.csv', 'w', newline='') as qrels_train_file:
        with open(args.output_data_dir + 'qrels_val.csv', 'w', newline='') as qrels_val_file:
            with open(args.output_data_dir + 'qrels_test.csv', 'w', newline='') as qrels_test_file:
                for index, row in qrels_df.iterrows():
                    if int(row['query_id']) in query_train_id:
                        csv_w = csv.writer(qrels_train_file)
                        if train_qrels_account == 0:
                            csv_w.writerow(['query_id', 'poi_id'])  # header
                        csv_w.writerow([int(row['query_id']), int(row['poi_id'])])
                        train_qrels_account += 1
                    elif int(row['query_id']) in query_val_id:
                        csv_w = csv.writer(qrels_val_file)
                        if val_qrels_account == 0:
                            csv_w.writerow(['query_id', 'poi_id'])  # header
                        csv_w.writerow([int(row['query_id']), int(row['poi_id'])])
                        val_qrels_account += 1
                    elif int(row['query_id']) in query_test_id:
                        csv_w = csv.writer(qrels_test_file)
                        if test_qrels_account == 0:
                            csv_w.writerow(['query_id', 'poi_id'])  # header
                        csv_w.writerow([int(row['query_id']), int(row['poi_id'])])
                        test_qrels_account += 1

    print(len(qrels_df))
    print("train_qrels_account: ", train_qrels_account)
    print("val_qrels_account: ", val_qrels_account)
    print("test_qrels_account: ", test_qrels_account)

    train_query_account = val_query_account = test_query_account = 0

    with open(args.output_data_dir + 'query_train.csv', 'w', newline='') as query_train_file:
        with open(args.output_data_dir + 'query_val.csv', 'w', newline='') as query_val_file:
            with open(args.output_data_dir + 'query_test.csv', 'w', newline='') as query_test_file:
                for index, row in query_df.iterrows():
                    if int(row['id']) in query_train_id:
                        csv_w = csv.writer(query_train_file)
                        if train_query_account == 0:
                            csv_w.writerow(['id', 'coor', 'text'])  # header
                        csv_w.writerow([int(row['id']), row['coor'], row['text']])
                        train_query_account += 1
                    elif int(row['id']) in query_val_id:
                        csv_w = csv.writer(query_val_file)
                        if val_query_account == 0:
                            csv_w.writerow(['id', 'coor', 'text'])  # header
                        csv_w.writerow([int(row['id']), row['coor'], row['text']])
                        val_query_account += 1
                    elif int(row['id']) in query_test_id:
                        csv_w = csv.writer(query_test_file)
                        if test_query_account == 0:
                            csv_w.writerow(['id', 'coor', 'text'])  # header
                        csv_w.writerow([int(row['id']), row['coor'], row['text']])
                        test_query_account += 1
                    else:
                        print("out of record: ", int(row['id']))
                        print("error exists")
    print("train_querys_account: ", train_query_account)
    print("val_querys_account: ", val_query_account)
    print("test_querys_account: ", test_query_account)
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

    args = parser.parse_args()

    return args

def main():
    args = get_arguments()
    preprocess(args)  
    

if __name__ == '__main__':
    main()