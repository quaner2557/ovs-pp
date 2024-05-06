from sklearn.cluster import KMeans
import numpy as np
import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset


# 假定EmbeddingDataset类已经定义
class EmbeddingDataset(Dataset):
    def __init__(self, json_files):
        self.data = {}
        for file_name in json_files:
            with open(file_name, 'r') as f:
                self.data.update(json.load(f))
        self.item_ids = list(self.data.keys())
        self.embeddings = np.array([self.data[item_id] for item_id in self.item_ids])

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        item_id = self.item_ids[idx]
        embedding = self.embeddings[idx]
        return item_id, embedding


# 定义一个函数来读取JSON文件并计算重复比例
def calculate_duplicate_ratio(filename):
    # 创建一个字典，用于存储编码及其出现次数
    code_count = {}

    # 读取JSON文件
    with open(filename, 'r') as file:
        data = json.load(file)

        # 遍历JSON对象中的每一个键值对
        for item_id, codes in data.items():
            # 将列表转换为元组（因为列表不可哈希，不能作为字典键）
            code_tuple = tuple(codes)

            # 统计编码出现的次数
            if code_tuple in code_count:
                code_count[code_tuple] += 1
            else:
                code_count[code_tuple] = 1

    # 计算重复比例
    total_items = len(code_count)
    duplicate_items = sum(1 for count in code_count.values() if count > 1)
    duplicate_ratio = duplicate_items / total_items if total_items > 0 else 0

    # 打印结果
    print(f"Total unique codes: {total_items}")
    print(f"Duplicate codes: {duplicate_items}")
    print(f"Duplicate ratio: {duplicate_ratio:.2%}")


json_files = [
    'datasets/asin_to_embedding_batch_0.json',
    'datasets/asin_to_embedding_batch_1.json',
    'datasets/asin_to_embedding_batch_2.json'
]
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = EmbeddingDataset(json_files)
data_loader = DataLoader(dataset, batch_size=1024, shuffle=False)

num_codebooks = 3  # 三个代码本
num_codes_per_book = 256  # 每个代码本256个代码
seed = 1234

# 提取所有嵌入和相应的item_ids
embeddings = dataset.embeddings
item_ids = dataset.item_ids

# 第一级全局K-Means聚类
kmeans_global = KMeans(n_clusters=num_codes_per_book, random_state=seed, n_init=2)
kmeans_global.fit(embeddings)
global_centers = kmeans_global.cluster_centers_
global_labels = kmeans_global.labels_

# 初始化codebooks和预测模型存储
second_level_codebooks = {}
third_level_codebooks = {}
third_level_predictors = {}

# 第二级聚类
for i in tqdm(range(num_codes_per_book), desc="Running 2nd-level KMeans"):
    mask = global_labels == i
    local_embeddings = embeddings[mask]
    local_num_clusters = min(len(local_embeddings), num_codes_per_book)
    if local_num_clusters > 1:
        kmeans_local = KMeans(n_clusters=local_num_clusters, random_state=seed, n_init=2)
        kmeans_local.fit(local_embeddings)
        second_level_codebooks[i] = (kmeans_local.cluster_centers_, kmeans_local.labels_, np.where(mask)[0])
    else:
        second_level_codebooks[i] = (np.array([]), np.array([]), np.array([]))

# 第三级聚类
for i in tqdm(second_level_codebooks.keys(), desc="Running 3rd-level KMeans"):
    centers, labels, global_indices = second_level_codebooks[i]
    third_level_codebooks[i] = {}
    third_level_predictors[i] = {}

    for j in set(labels):
        sub_indices = np.where(labels == j)[0]
        local_embeddings = embeddings[global_indices[sub_indices]]

        if len(local_embeddings) > 1:
            kmeans_third = KMeans(n_clusters=min(len(local_embeddings), num_codes_per_book), random_state=seed,
                                  n_init=2)
            kmeans_third.fit(local_embeddings)
            third_level_codebooks[i][j] = kmeans_third.cluster_centers_
            third_level_predictors[i][j] = kmeans_third  # 存储模型以备后用
        else:
            third_level_codebooks[i][j] = local_embeddings if len(local_embeddings) > 0 else np.array([])
            third_level_predictors[i][j] = None  # 如果没有数据或只有一个中心，设置为None

# 对每个嵌入进行编码
encoded_items = {}
for i, item_id in tqdm(enumerate(item_ids), desc="Encoding items", total=len(item_ids)):
    current_embedding = embeddings[i]

    # 第一级编码
    global_code = global_labels[i]

    # 第二级编码
    centers, labels, global_indices = second_level_codebooks[global_code]
    if len(centers) == 0:
        encoded_items[item_id] = [global_code, -1, -1]
        continue

    second_level_index = np.where(global_indices == i)[0][0]
    second_level_code = labels[second_level_index]

    # 第三级编码
    third_level_centers_dict = third_level_codebooks.get(global_code, {})
    third_level_centers = third_level_centers_dict.get(second_level_code, np.array([]))
    if len(third_level_centers) == 0:
        encoded_items[item_id] = [global_code, second_level_code, -1]
        continue

    kmeans_third = third_level_predictors[global_code][second_level_code]
    if kmeans_third:
        third_level_code = kmeans_third.predict([current_embedding])[0]
    else:
        third_level_code = -1

    # 保存编码结果
    encoded_items[item_id] = [global_code, second_level_code, third_level_code]

# 将编码结果转换为JSON可序列化的格式
encoded_items_int = {item_id: [int(c) for c in codes] for item_id, codes in encoded_items.items()}

# 将编码结果保存到JSON文件
with open('datasets/encoded_items.json', 'w') as f:
    json.dump(encoded_items_int, f)

print("Items have been encoded and saved to 'encoded_items.json'")
# 统计编码的重合度
calculate_duplicate_ratio('datasets/encoded_items.json')