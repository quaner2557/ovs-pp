import json
from datetime import datetime


def split_data(user_item_file):
    # 读取原始数据文件
    with open(user_item_file, 'r') as f:
        user_item_dict = json.load(f)

    test_dict = {}
    valid_dict = {}

    # 对每个用户对应的商品列表进行处理
    for user, items in user_item_dict.items():
        # 确保items长度不会超过20
        items = items[:20]

        if len(items) >= 3:
            # 如果有3个以上的项目，则从第三个项目到倒数第二个项目作为验证集
            valid_items = items[-min(20, len(items) - 1):-1]
            valid_dict[user] = valid_items
            # 如果有3个以上的项目，则从第二个项目到最后一个项目作为测试集
            test_items = items[-min(20, len(items)):]
            test_dict[user] = test_items
        else:
            continue

    # 将处理后的数据保存到指定的JSON文件中
    with open('datasets/test.json', 'w') as f:
        json.dump(test_dict, f, indent=4)

    with open('datasets/valid.json', 'w') as f:
        json.dump(valid_dict, f, indent=4)


# 准备一个空字典来存储结果
reviewer_dict = {}

# 打开你的json文件并逐行读取
with open('datasets/reviews_Beauty_5.json', 'r') as file:
    for line in file:
        # 将json字符串解析成字典
        review = json.loads(line)

        # 获取需要的字段
        reviewerID = review['reviewerID']
        asin = review['asin']
        reviewTime = review['reviewTime']

        # 将日期字符串转换为datetime对象
        reviewDate = datetime.strptime(reviewTime, '%m %d, %Y')

        # 如果reviewerID已经在字典中，追加asin和reviewTime的元组
        if reviewerID in reviewer_dict:
            reviewer_dict[reviewerID].append((asin, reviewDate))
        else:
            # 否则，为这个reviewerID创建新条目
            reviewer_dict[reviewerID] = [(asin, reviewDate)]

# 根据reviewTime对asin列表进行排序
for reviewerID in reviewer_dict:
    # 对每个reviewerID的列表按reviewDate排序
    reviewer_dict[reviewerID].sort(key=lambda x: x[1])

    # 去除reviewDate，只保留asin
    reviewer_dict[reviewerID] = [asin for asin, reviewDate in reviewer_dict[reviewerID]]

# 将结果字典保存到本地json文件
with open('datasets/reviewer_asin_dict.json', 'w') as outfile:
    json.dump(reviewer_dict, outfile)

print('字典已保存到reviewer_asin_dict.json文件中。')

user_item_file = 'datasets/reviewer_asin_dict.json'
split_data(user_item_file)
print('文件已切分成训练和测试集')