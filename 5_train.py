# 依赖包
import torch
from torch.utils.data import DataLoader, Dataset
import json
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from collections import deque
import os
from random import randint

# 使用t5-small预训练模型
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

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
        with open(item_encode_file, 'r') as f:
            self.item_encode_dict = json.load(f)

        self.tokenizer = tokenizer

        # 处理编码冲突
        self.resolve_encoding_conflicts()
        self.add_token()

        # 自定义token编码
        self.token_encode = {}

        # 准备数据
        self.sequences = []
        self.targets = []
        self.user_ids = []
        for user_id, item_ids in self.user_item_dict.items():
            self.user_ids.append(user_id)

        # 用 dict 替换 LabelEncoder
        self.user_ids = list(self.user_item_dict.keys())
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}

        self.token_cache = {}  # 缓存编码结果以避免重复编码相同的token

        self.running()

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
                             item in self.item_encode_dict]
            n_items = len(encoded_items)

            for i in range(n_items - 1):  # 确保有至少两个item，留一个作为target
                # 随机选择序列长度，确保至少有1个item并且不超过max_sequence_length和当前items数量-1
                seq_len = randint(1, min(self.sequence_length, n_items - i - 1))
                sequence = encoded_items[i:i + seq_len]
                target = encoded_items[i + seq_len]  # 选取序列后的一个item作为预测目标

                flattened_sequence = [user_token] + [item for sublist in sequence for item in sublist]
                self.sequences.append(flattened_sequence)
                self.targets.append(target)

    def resolve_encoding_conflicts(self):
        self.max_item_id = 0
        # 创建一个字典来记录每个前三位编码的出现次数
        prefix_counts = defaultdict(int)
        # 字典，将前三位编码映射到它们的新第四位编码
        prefix_to_new_code = defaultdict(lambda: 0)

        # 遍历所有items，统计前三位编码的出现次数
        for code in self.item_encode_dict.values():
            prefix = tuple(code[:3])  # 前三位编码作为key
            prefix_counts[prefix] += 1

        # 遍历所有items，根据出现次数分配新的第四位编码
        for item, code in self.item_encode_dict.items():
            prefix = tuple(code[:3])
            if prefix_counts[prefix] == 1:
                # 如果前三位编码唯一，那么第四位就是0
                self.item_encode_dict[item].append(0)
            else:
                # 如果前三位编码不唯一，分配一个新的唯一编码
                self.max_item_id = max(self.max_item_id, prefix_to_new_code[prefix])
                self.item_encode_dict[item].append(prefix_to_new_code[prefix])
                prefix_to_new_code[prefix] += 1

    def add_token(self):
        # 使用Transformers库向T5 tokenizer中添加新的特殊token
        self.tokenizer.add_tokens(['<user_id_{}>'.format(i) for i in range(2001)])
        self.tokenizer.add_tokens(['<item_id_{}>'.format(i) for i in range(3 * 256 + self.max_item_id + 1)])
        print('self.max_item_id is:', self.max_item_id)

        # 调整模型的token嵌入层大小以匹配更新后的tokenizer大小
        model.resize_token_embeddings(len(self.tokenizer))

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

# 创建数据集
user_item_file = 'datasets/valid.json'
item_encode_file = 'datasets/encoded_items.json'
dataset = RecommenderDataset(user_item_file, item_encode_file)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=4)


# 学习率
init_learning_rate = 5e-05
num_warmup_steps = 0
global_step = 0  # 训练200k步

# 定义学习率计划
def custom_lr_scheduler(step):
    if step <= num_warmup_steps:
        return init_learning_rate
    else:
        return init_learning_rate * ((step-num_warmup_steps) ** -0.5)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=init_learning_rate)
accumulation_steps = 4
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 移动模型到指定设备
model.to(device)
num_training_steps_per_epoch = len(data_loader)
print('num_training_steps_per_epoch is:',num_training_steps_per_epoch)

max_step = 200 * num_training_steps_per_epoch

# 训练代码
model.train()

# 初始化损失队列
recent_losses = deque(maxlen=10)

while global_step < max_step:
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Training")
    for step, (input_sequences, target_tests, attention_masks) in progress_bar:
        current_lr = init_learning_rate
        for g in optimizer.param_groups:
            g['lr'] = current_lr

        input_sequences = input_sequences.to(device)
        target_tests = target_tests.to(device)
        attention_masks = attention_masks.to(device)

        # Forward pass
        outputs = model(input_ids=input_sequences, labels=target_tests, attention_mask=attention_masks)
        loss = outputs.loss / accumulation_steps
        loss.backward()

        if step % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 4

            # 添加最新的损失到队列中，不管当前步数是什么
            recent_losses.append(loss.item())

            # 每1000步打印一次平均损失
            if step > 0 and step % 100 == 0:
                avg_loss = sum(recent_losses) / len(recent_losses)
                print(f"Step {global_step}: Last 10 steps' Average Loss {avg_loss}")

        # 每个epoch结束时保存模型和优化器的状态
        if global_step % num_training_steps_per_epoch in [0, 1, 2, 3]:
            # 保存模型
            checkpoint_path = os.path.join('checkpoints',
                                           f"t5_small_{(global_step + 1) // num_training_steps_per_epoch}")
            model.save_pretrained(checkpoint_path)
            # 保存tokenizer
            tokenizer.save_pretrained(checkpoint_path)
            # 保存优化器的状态
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, 'optimizer.pt'))
            with open(os.path.join(checkpoint_path, 'user_id_to_index.json'), 'w') as f:
                json.dump(dataset.user_id_to_index, f)
            with open(os.path.join(checkpoint_path, 'item_encode_dict.json'), 'w') as f:
                json.dump(dataset.item_encode_dict, f)

        # Break the loop if max step is reached
        if global_step >= max_step:
            break