# 先清洗数据
import json
import tqdm


def replace_ith_quote_after_keyword(input_string, keyword, ith):
    # 查找关键字在字符串中的位置
    keyword_pos = input_string.find(keyword)
    if keyword_pos == -1:
        # 如果关键字不在字符串中，返回原始字符串
        return input_string

    # 从关键字位置开始查找第 ith 个双引号
    quote_count = 0
    for i in range(keyword_pos, len(input_string)):
        if input_string[i] == '"':
            quote_count += 1
            if quote_count == ith:
                # 找到第 ith 个双引号，替换成单引号
                return input_string[:i] + "'" + input_string[i + 1:]

    # 如果找不到足够的双引号，返回原始字符串
    return input_string


# 关键字
keyword = "description"
# 指定要替换的是第几个双引号

# 读取JSON文件，并处理每一行
with open('datasets/meta_Beauty.json', 'r') as file:
    # 打开两个文件：一个用于存储成功解析的行，另一个用于存储错误
    with open('datasets/processed_lines.json', 'w') as success_file, open('datasets/error_lines.json',
                                                                          'w') as error_file:
        for line in tqdm.tqdm(file, desc='Processing meta_Beauty.json'):
            # 对每一行应用替换逻辑
            tmp_line = line.replace('"', "'").replace("\\n", " ").replace("\\'", "'").replace("': '", '": "').replace(
                "', '", '", "').replace("']", '"]').replace("['", '["').replace("':", '":').replace("{'", '{"').replace(
                "'}", '"}').replace(", '", ', "').replace('\', \"title\"', '\", \"title\"').replace("\n",
                                                                                                    "\\n").replace("\r",
                                                                                                                   "\\r").replace(
                "\t", "\\t").strip("\\n")
            # tmp_line = line
            try:
                # 尝试将修改后的字符串解析为JSON
                data = json.loads(tmp_line)
                # 写入成功解析的行到相应文件
                success_file.write(tmp_line + '\n')
            except Exception as e:
                # 写入导致错误的行到相应文件
                try:
                    ith_quote = 3
                    new_line = replace_ith_quote_after_keyword(tmp_line, keyword, ith_quote)
                    data = json.loads(new_line)
                    # 写入成功解析的行到相应文件
                    success_file.write(new_line + '\n')
                except Exception as e:
                    try:
                        ith_quote = 3
                        new_line = replace_ith_quote_after_keyword(tmp_line, keyword, ith_quote)
                        data = json.loads(new_line)
                        # 写入成功解析的行到相应文件
                        success_file.write(new_line + '\n')
                    except Exception as e:
                        error_file.write(new_line + '\n')

print('step1 finished!')