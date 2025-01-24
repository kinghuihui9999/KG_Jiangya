from config import *
from utils import *
from model import *

if __name__ == '__main__':
    id2label, _ = get_label()

    model = TextCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR + 'model_5.pth', map_location=DEVICE))

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    # text = '变压器上面都有什么其他设备？'
    # text = '怎么解决变压器失灵的问题？'
    text = '玻璃损坏会造成什么？'
    tokened = tokenizer(text)
    input_ids = tokened['input_ids']
    mask = tokened['attention_mask']
    if len(input_ids) < TEXT_LEN:
        pad_len = (TEXT_LEN - len(input_ids))
        input_ids += [BERT_PAD_ID] * pad_len
        mask += [0] * pad_len

    pred = model(torch.tensor([input_ids]).to(DEVICE), torch.tensor([mask]).to(DEVICE))

    max_value, max_key = torch.max(pred, dim=1)

    print(max_value[0].item(), max_key[0].item(), id2label[max_key[0].item()])