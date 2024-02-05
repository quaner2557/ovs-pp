# 参照论文
# we use item's content features such as title, price, brand, and category to construct a sentence
# which is then passed to the pre-trained Sentence-T5 model to obtain the item's semantic embedding of 768 dimension
import json
import tqdm
import torch
from transformers import T5Model, T5Tokenizer


# item_info的类定义
class ItemInfo:
    def __init__(self, itemID, title, description, brand, category):
        self.itemID = itemID
        self.title = title
        self.description = description
        self.brand = brand
        self.category = category

    def to_sentence(self):
        return f"itemID: {self.itemID} ;title: {self.title} ;Description: {self.description};category: {self.category};brand: {self.brand}"


# 定义一个函数来处理商品并生成其语义嵌入
# 修改后的get_semantic_embeddings函数，以处理一个批次的ItemInfo对象
def get_semantic_embeddings(item_infos, model, tokenizer, device):
    """
    返回：包含批次中每个商品768维语义嵌入的列表
    """
    # 构造输入句子列表
    sentences = [item_info.to_sentence() for item_info in item_infos]
    # 编码句子列表
    inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    encoder_outputs = model.encoder(**inputs)
    last_hidden_states = encoder_outputs.last_hidden_state
    sentence_embeddings = torch.mean(last_hidden_states, dim=1)
    return sentence_embeddings


if __name__ == '__main__':
    device = torch.device("cuda")

    asin_to_embedding = {}
    batch_size = 8  # 可以根据实际情况调整批大小

    # 初始化模型和分词器
    model_name = "sentence-transformers/sentence-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5Model.from_pretrained(model_name)
    model.to(device)  # 将模型移动到GPU

    # ...
    with open('datasets/meta_Beauty.json', 'r') as file:
        batch = []
        for line in tqdm.tqdm(file, desc='Processing meta_Beauty.json'):
            # 将每一行的JSON字符串转换成字典
            try:
                data = json.loads(
                    line.replace('"', "'").replace("\\n", " ").replace("\\'", "'").replace("': '", '": "').replace("', '", '", "').replace("']", '"]').replace("['", '["').replace("':", '":').replace("{'", '{"').replace(", '", ', "'))
            except:
                # print(line)
                continue

            # 构造ItemInfo对象
            itemID = data.get('asin')
            title = data.get('title')
            description = data.get('description')
            try:
                category = ','.join(data.get('categories')[0])
            except:
                category = ''
            brand = data.get('brand')
            item_info = ItemInfo(itemID, title, description, brand, category)
            batch.append(item_info)

            if len(batch) == batch_size:
                sentence_embeddings = get_semantic_embeddings(batch, model, tokenizer, device)
                for item, embedding in zip(batch, sentence_embeddings):
                    # 将embedding转换为列表，以便能够序列化为JSON
                    asin_to_embedding[item.itemID] = embedding.tolist()
                batch = []  # 清空批处理列表

        # 不要忘记处理最后一个不完整的批次
        if batch:
            sentence_embeddings = get_semantic_embeddings(batch, model, tokenizer, device)
            for item, embedding in zip(batch, sentence_embeddings):
                asin_to_embedding[item.itemID] = embedding.tolist()

    # ...
    with open('asin_to_embedding.json', 'w') as outfile:
        print(len(asin_to_embedding))
        json_str = json.dumps(asin_to_embedding, indent=4)
        outfile.write(json_str)