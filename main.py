
# we use item's content features such as title, price, brand, and category to construct a sentence
# which is then passed to the pre-trained Sentence-T5 model to obtain the item's semantic embedding of 768 dimension

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 初始化T5模型和分词器
model_name = "t5-base"  # 或者其他预训练的Sentence-T5模型名称
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# item_info的类定义
class ItemInfo:
    def __init__(self, title, price, brand, category):
        self.title = title
        self.price = price
        self.brand = brand
        self.category = category

# 定义一个函数来处理商品并生成其语义嵌入
def get_semantic_embedding(item_info):
    """
    item_info: dict 包含商品信息如 {'title': '商品标题', 'price': '商品价格', 'brand': '品牌名', 'category': '商品类别'}

    返回：torch.tensor 商品的768维语义嵌入
    """
    # 构造输入句子
    sentence = f"title: {item_info['title']} ;price: {item_info['price']} ;brand: {item_info['brand']} ;category: {item_info['category']}"

    # 对句子进行编码
    inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=512, return_tensors="pt")

    # 获取隐状态（最后一层的输出）作为语义嵌入
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state[:, -1, :]  # 取最后一个时间步的向量作为语义嵌入

    return last_hidden_state

# 示例使用
item_example = {
    'title': 'iPhone 13 Pro Max',
    'price': '9999元',
    'brand': 'Apple',
    'category': '手机数码'
}

embedding = get_semantic_embedding(item_example)
# print embedding
print(f"商品的语义嵌入：{embedding}")
print(f"商品的语义嵌入维度：{embedding.shape}")

