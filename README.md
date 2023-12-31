# LIST


### Code for paper 'LIST: Learning to Index Spatio-Textual Data for Dense Vectors based Spatial Keyword Queries'  

LIST is a novel solution, which learns to index spatio-textual data for answering dense vectors based spatial keyword queries effectively and efficently. It is featured with two novel conponents: First, it propose a lightweight and effective relevance model, which is based on dense retrieval models. Secondly, we introduce a novel machine learning based Approximate Nearest Neighbor Search (ANNS) index, which utilizes the learning-to-cluster technique to group relevant queries and objects together while separating irrelevant queries and objects.


#### The full version paper can be downloaded [here](https://drive.google.com/drive/folders/1oNRVD1Z5fVH_vjJAHY_uSGYVQR2Ss-cd).

### Requirements

* Python 3.8
* Pytorch 2.0.1
* transformers 4.10.2


### LIST
The following image depicts the workflow of LIST, which has three phases:  the training, indexing, and query phase. During the training phase, we first train our relevance model, and then we train our index.  During the indexing phase, each object is partitioned into one cluster. Note that our index is not limited to our proposed relevance model, and can also be applied for other embedding-based spatial-textual relevance models. During the query phase, the trained index routes each query to a cluster with the highest probability. In the cluster, k objects with the highest scores ranked by the trained relevance model are returned for the query.

<img src="imgs/framework.png" alt="The three phases of our retriever LIST: the training, indexing, and query phase. LIST consists of a relevance model (shown in green) and an index (shown in yellow)." width="90%"/>


### Datasets

We evaluate on the Beijing, Shanghai and Geo-Glue dataset.


### Testing

To test LIST-R on the Geo-Glue dataset.

The checkpoint of trained relevance model and trained index can be downloaded [here](https://drive.google.com/drive/folders/1oNRVD1Z5fVH_vjJAHY_uSGYVQR2Ss-cd).


```
python3 -m src.rank.inference  --checkpoint_file ./checkpoints/rank/geo-glue/checkpoint-file \
        --model_type bert-base-chinese --spatial_step_k 1000 --dataset geo-glue\
        --origin_data_dir ./data/geo-glue/processed_data/ \
        --output_dir ./data/geo-glue/processed_data/ \
        --max_poi_length 96 --max_query_length 32
```
```
python3 -m src.rank.brute_search  --checkpoint_file ./checkpoints/rank/geo-glue/checkpoint-file \
        --model_type bert-base-chinese --spatial_step_k 1000 --dataset geo-glue\
        --origin_data_dir ./data/geo-glue/processed_data/ \
        --output_dir ./data/geo-glue/processed_data/ \
        --embedding_dir ./data/geo-glue/processed_data/
```

To test LIST on the Geo-Glue dataset.

```
python3 -m src.cluster.inference_l2c  --rank_checkpoint_file ./checkpoints/rank/geo-glue/checkpoint-file \
        --cluster_checkpoint_file  ./checkpoints/cluster/geo-glue/checkpoint-100-0.3053 \
        --model_type bert-base-chinese --spatial_step_k 1000 --dataset geo-glue\ 
        --num_cluster 300 --hidden_dim 768 --num_layers 2 --min_precision 0\
        --origin_data_dir ./data/geo-glue/processed_data/ \
        --embedding_dir ./data/geo-glue/processed_data/ \
```

