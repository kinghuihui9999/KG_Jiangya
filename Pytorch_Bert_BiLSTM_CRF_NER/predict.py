from utils import *
from model import *
from config import *

if __name__ == '__main__':
    # text = '橡胶水导轴承。橡胶水导轴承结构比较简单，主要部件包括轴承体、润滑水箱、轴封、橡胶轴瓦等。'
    # text = '活动导叶立面密封缺陷可能的原因：（2）导叶压紧行程不够。导叶压紧行程不够使导叶立面密封未压紧，导致蜗壳内的高压水流冲击转轮，使机组停机困难或者停机后蠕动。'
    # text = '宝马发动机过热的原因是电压不足，油温过高'
    text = '压板螺栓会发生哪些故障'
    # text = '调压闸门故障。处理方法：修复或更换受损的调压闸门，确保操作机构正常运行。对于液压系统问题，进行检修或更换故障元件。进行全面的调试和测试，确保调压闸门能够准确调节水流。'
    # text = '发电机转子是产生磁场、变换能量和传递扭矩的转动部件，是组成发电机通风系统的主要结构要素，是水轮发电机重要的组成部件之一。'
    # text = '集电环打火的原因有以下几种:电刷磨损过多，恒压弹簧压力不适，集电环表面表面粗糙度不够，电刷的材质不均匀，受油器渗油。'
    # input = torch.tensor([[word2id.get(w, WORD_UNK_ID) for w in text]])
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    input = torch.tensor([tokenizer.encode(list(text), add_special_tokens=False)])
    mask = torch.tensor([[1] * len(text)]).bool()

    model = Model()
    model.load_state_dict(torch.load(MODEL_DIR + '0211_13_30.pth', map_location=DEVICE))

    y_pred = model(input, mask)
    id2label, _ = get_label()

    label = [id2label[l] for l in y_pred[0]]
    print(text)
    # print(label)

    info = extract(label, text)
    print(info)