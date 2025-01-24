import ast

# 假设你的文件路径是 'path/to/your/file.txt'
file_path = './output/final_results.txt'

# 读取文件
with open(file_path, 'r',encoding='utf-8') as file:
    file_content = file.read()

# 解析内容为列表
# 使用 ast.literal_eval 安全地将字符串转换为列表
my_list = ast.literal_eval(file_content)

# 使用 set 删除重复项
unique_set = set(my_list)

# 如果需要，将 set 转换回 list
unique_list = list(unique_set)

print(unique_list)


