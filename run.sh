#!/bin/bash

GZIPPED_FILE_META="datasets/meta_Beauty.json.gz"
GZIPPED_FILE_REVIEWS="datasets/reviews_Beauty_5.json.gz"

# 解压json.gz文件
gunzip -k $GZIPPED_FILE_META
gunzip -k $GZIPPED_FILE_REVIEWS

# 运行Python清洗数据的脚本
python 1_clean_data.py
python 2_embedding.py
python 3_kmeans.py
python 4_split_data.py
python 5_train.py
python 6_test.py