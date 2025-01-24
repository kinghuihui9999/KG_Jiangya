from utils import *
from model import *
from config import *

if __name__ == '__main__':
    dataset = Dataset('test')
    loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    with torch.no_grad():
        model = Model()
        model.load_state_dict(torch.load(MODEL_DIR + '0211_13_30.pth', map_location=DEVICE))

        y_true_list = []
        y_pred_list = []

        id2label, _ = get_label()

        for b, (input, target, mask) in enumerate(loader):

            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            y_pred = model(input, mask)
            loss = model.loss_fn(input, target, mask)

            print('>> batch:', b, 'loss:', loss.item())
        
            # # 拼接返回值
            # for lst in y_pred:
            #     y_pred_list += lst
            # for y,m in zip(target, mask):
            #     y_true_list += y[m==True].tolist()

            for lst in y_pred:
                y_pred_list.append([id2label[i] for i in lst])
            for y,m in zip(target, mask):
                y_true_list.append([id2label[i] for i in y[m==True].tolist()])

        print(report(y_true_list, y_pred_list, output_dict=False))


        # 整体准确率
        # y_true_tensor = torch.tensor(y_true_list)
        # y_pred_tensor = torch.tensor(y_pred_list)
        # accuracy = (y_true_tensor == y_pred_tensor).sum()/len(y_true_tensor)
        # print('>> total:', len(y_true_tensor), 'accuracy:', accuracy.item())
        