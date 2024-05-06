# 依赖包
import torch
from torch.utils.data import DataLoader, Dataset
import json
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Model, AdamW
import os

# 设置checkpoint路径。这应当与保存模型时相同的路径
checkpoint_path = "checkpoints/t5_small_200"  #修改使用的checkpoint

# 加载tokenizer
tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)

# 加载模型
model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)

# 定义 collate_fn 来动态填充序列并生成attention mask
def collate_fn(batch):
    inputs, targets_test = zip(*batch)
    # 填充 inputs 使它们具有相同的长度，并生成相应的attention masks
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = torch.tensor([[float(i > 0) for i in seq] for seq in padded_inputs])
    # 填充 targets_val 和 targets_test，target的padding_value改成-100
    padded_targets_test = pad_sequence(targets_test, batch_first=True, padding_value=tokenizer.pad_token_id)
    return padded_inputs, padded_targets_test, attention_masks


class RecommenderDataset(Dataset):
    def __init__(self, user_item_file, item_encode_file, sequence_length=20):
        self.sequence_length = sequence_length
        # 加载数据
        with open(user_item_file, 'r') as f:
            self.user_item_dict = json.load(f)

        self.tokenizer = tokenizer

        # 从json文件中读取数据到字典
        with open(os.path.join(checkpoint_path, 'user_id_to_index.json'), 'r') as f:
            self.user_id_to_index = json.load(f)

        with open(os.path.join(checkpoint_path, 'item_encode_dict.json'), 'r') as f:
            self.item_encode_dict = json.load(f)

        # 准备数据
        self.sequences = []
        self.targets = []
        self.cal_dict1 = {}
        self.cal_dict2 = {}
        self.cal_dict3 = {}
        self.cal_dict4 = {}
        self.running()
        self.calculate()

    def get_all_encoded_items(self):
        # 确保我们有一个编码后的商品ID的列表
        all_encoded_items = []
        for code in self.item_encode_dict.values():
            # 用正确的方式添加商品编码
            # 这里假设编码是以四元组形式存储的。
            if len(code) == 4:
                all_encoded_items.append(code)
        return all_encoded_items

    def running(self):
        for user_id, item_ids in tqdm(self.user_item_dict.items(), desc="Processing users"):
            user_index = self.user_id_to_index[user_id]
            user_token = user_index % 2000  # Hashing Trick
            encoded_items = [self.convert_token(self.item_encode_dict[item]) for item in item_ids if
                             item in self.item_encode_dict]  # [seq_len,4]
            sequence = encoded_items[:-1]  # 最后一个为test
            self.sequences.append(self.convert_token([user_token]) + [item for sublist in sequence for item in sublist])
            self.targets.append(encoded_items[-1:])  # test item

    def convert_token(self, encode):
        assert len(encode) in (1, 4)
        lists = []
        if len(encode) == 4:
            lists = ['<item_id_{}>'.format(encode[i] + i * 256) for i in range(len(encode))]
        if len(encode) == 1:
            lists = ['<user_id_{}>'.format(encode[i]) for i in range(len(encode))]
        # 使用rstrip()方法删除字符串末尾的空白字符，确保没有多余的空格
        items_str = ''.join(lists).rstrip()
        # 使用tokenizer将字符串编码为数字ID
        input_ids = self.tokenizer.encode(items_str, add_special_tokens=True)
        return input_ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        target_test = self.targets[index]

        # 如果 sequence 是列表的列表，则展平它。否则直接使用 sequence。
        if all(isinstance(sublist, list) for sublist in sequence):
            sequence_flat = [item for sublist in sequence for item in sublist]
        else:
            sequence_flat = sequence

        # 确保 target_test 是列表
        target_test = target_test if isinstance(target_test, list) else [target_test]

        # 返回 tensor
        return torch.tensor(sequence_flat, dtype=torch.long), torch.tensor(target_test, dtype=torch.long).squeeze()

    def calculate(self):

        for target in self.targets:
            if target[0][0] not in self.cal_dict1:
                self.cal_dict1[target[0][0]] = 1
            else:
                self.cal_dict1[target[0][0]] += 1
            if target[0][1] not in self.cal_dict2:
                self.cal_dict2[target[0][1]] = 1
            else:
                self.cal_dict2[target[0][1]] += 1
            if target[0][2] not in self.cal_dict3:
                self.cal_dict3[target[0][2]] = 1
            else:
                self.cal_dict3[target[0][2]] += 1
            if target[0][3] not in self.cal_dict4:
                self.cal_dict4[target[0][3]] = 1
            else:
                self.cal_dict4[target[0][3]] += 1
        self.cal_dict1 = sorted(self.cal_dict1.items(), key=lambda item: item[1], reverse=True)
        self.cal_dict2 = sorted(self.cal_dict2.items(), key=lambda item: item[1], reverse=True)
        self.cal_dict3 = sorted(self.cal_dict3.items(), key=lambda item: item[1], reverse=True)
        self.cal_dict4 = sorted(self.cal_dict4.items(), key=lambda item: item[1], reverse=True)


# 创建数据集
user_item_file = 'datasets/test.json'
item_encode_file = 'datasets/encoded_items.json'
dataset = RecommenderDataset(user_item_file, item_encode_file)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=4)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 移动模型到指定设备
model.to(device)

# 预测代码
model.eval()
hit_count, total_count = 0, 0
progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Testing")
for step, (input_sequences, target_tests, attention_masks) in progress_bar:
    input_sequences = input_sequences.to(device)
    target_tests = target_tests.to(device)
    attention_masks = attention_masks.to(device)

    # Forward pass
    outputs = model.generate(input_ids=input_sequences, attention_mask=attention_masks, num_beams=5,
                             num_return_sequences=5)

    for i, target_test in enumerate(target_tests):
        # Assuming that target_tests are raw text for simplicity here, you might need to convert IDs to text
        target_text = tokenizer.decode(target_test, skip_special_tokens=True)
        # Check if any of the generated sequences for this item contains the target
        hit = any(target_text == tokenizer.decode(generated, skip_special_tokens=True) for generated in outputs[i])
        if hit:
            hit_count += 1

        total_count += 1

    progress_bar.set_postfix(hit_rate=hit_count / total_count)

# 计算hitrate
hit_rate = hit_count / total_count

print("Total Hit Rate: {:.2f}%".format(100 * hit_rate))