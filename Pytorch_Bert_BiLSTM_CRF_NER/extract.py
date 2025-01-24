from utils import *
from model import *
from config import *
import ast
from extract import *

def process_file(input_file_path, output_file_path):
    # 加载模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    model = Model()
    model.load_state_dict(torch.load(MODEL_DIR + 'bert_bilstm_crf_model.pth', map_location=DEVICE))
    id2label, _ = get_label()

    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
         open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            text = line.strip()
            if not text:
                continue  # 跳过空行
            input_ids = torch.tensor([tokenizer.encode(list(text), add_special_tokens=False)])
            mask = torch.tensor([[1] * len(input_ids[0])]).bool()

            # 进行预测
            y_pred = model(input_ids, mask)
            label = [id2label[l] for l in y_pred[0]]

            # 抽取信息
            info = [extract(label,text)]
            flattened_info = [item[1] for sublist in info for item in sublist]

            # 将信息写入到输出文件，这里假设info是字符串
            output_file.write(f"{flattened_info}\n")

# Initialize an empty list to hold all elements
merged_list = []

# Open and read the file line by line
def results_process(output_file_path,out_file_path):
    with open(output_file_path, 'r', encoding='utf-8') as output_file,\
        open(out_file_path, 'w', encoding='utf-8') as out_file:
            for line in output_file:
                # Remove newline characters and any potential leading/trailing spaces
                line = line.strip()
                if line:  # Check if the line is not empty
                    try:
                        # Convert string representation of list to actual list
                        line_list = ast.literal_eval(line)
                        if isinstance(line_list, list):  # Ensure the parsed object is a list
                            merged_list.extend(line_list)
                    except ValueError as e:
                        print(f"Error processing line: {line} with error {e}")
            out_file.write(str(merged_list))

# 调用函数处理文件
input_file_path = './input/data/merge.txt'  # 替换为你的输入文件路径
output_file_path = './output/results.txt'  # 替换为你想写入结果的文件路径
out_file_path = './output/final_results.txt'
process_file(input_file_path, output_file_path)
results_process(output_file_path,out_file_path)
# print(merged_list)




