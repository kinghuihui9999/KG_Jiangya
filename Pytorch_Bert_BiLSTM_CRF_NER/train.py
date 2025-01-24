from utils import *
from model import *
from config import *
from model_dev import dev

if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    best_f1_score = 0  # 初始化最高F1分数为0
    best_epoch = -1  # 初始化最佳epoch

    model = Model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for e in range(EPOCH):
        for b, (input, target, mask) in enumerate(loader):

            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            y_pred = model(input, mask)

            loss = model.loss_fn(input, target, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if b % 10 == 0:
            #     print('>> epoch:', e, 'loss:', loss.item())

        dev_report = dev(model)

        f1_score = round(dev_report['macro avg']['f1-score'], 4)
        precision = round(dev_report['macro avg']['precision'], 4)  # 获取精确度值
        recall = round(dev_report['macro avg']['recall'], 4)  # 获取召回率值
        print(f'>> epoch: {e}, loss: {loss.item()}, dev_precision: {precision}, dev_recall: {recall}, dev_f1: {f1_score}')


        # 保存模型参数
        if f1_score > best_f1_score:
            best_f1_score = f1_score  # 更新最高F1分数
            best_epoch = e  # 更新最佳epoch
            # 保存当前模型的参数
            torch.save(model.state_dict(), MODEL_DIR + f'best_model.pth')
            # 最后，你可以打印出获得最高F1分数的epoch和分数
    print(f"Best F1 score of {best_f1_score} was achieved at epoch {best_epoch}")
