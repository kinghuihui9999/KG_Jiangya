from config import *
from utils import *
from model import *
from model_dev import dev

# if __name__ == '__main__':
#     id2label, _ = get_label()
#
#     train_dataset = Dataset('train')
#     train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#
#     model = TextCNN().to(DEVICE)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#     loss_fn = nn.CrossEntropyLoss()
#     best_f1_score = 0  # 初始化最高F1分数为0
#     best_epoch = -1  # 初始化最佳epoch
#
#     for e in range(EPOCH):
#         for b, (input, mask, target) in enumerate(train_loader):
#             input = input.to(DEVICE)
#             mask = mask.to(DEVICE)
#             target = target.to(DEVICE)
#
#             pred = model(input, mask)
#             loss = loss_fn(pred, target)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             y_pred = torch.argmax(pred, dim=1)
#             report = evaluate(y_pred.cpu().data.numpy(), target.cpu().data.numpy(), id2label, output_dict=True)
#
#         dev_report = dev(model)
#
#         f1_score = round(dev_report['macro avg']['f1-score'], 4)
#         precision = round(dev_report['macro avg']['precision'], 4)  # 获取精确度值
#         recall = round(dev_report['macro avg']['recall'], 4)  # 获取召回率值
#         print(f'>> epoch: {e}, loss: {loss.item()}, dev_precision: {precision}, dev_recall: {recall}, dev_f1: {f1_score}')
#
#         if f1_score > best_f1_score:
#             best_f1_score = f1_score  # 更新最高F1分数
#             best_epoch = e  # 更新最佳epoch
#             # 保存当前模型的参数
#             torch.save(model.state_dict(), MODEL_DIR + f'best_model.pth')
#             # print(f"Saved better model with F1: {best_f1_score} at epoch: {best_epoch}")
#
#             # 最后，你可以打印出获得最高F1分数的epoch和分数
#     print(f"Best F1 score of {best_f1_score} was achieved at epoch {best_epoch}")

from config import *
from utils import *
from model import TextCNN
from model_dev import dev

if __name__ == '__main__':
    id2label, _ = get_label()

    # 加载训练数据
    train_dataset = Dataset('train', vocab=vocab, sample_path=TRAIN_SAMPLE_PATH)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    model = TextCNN(VOCAB_SIZE, EMBEDDING_DIM, NUM_FILTERS, FILTER_SIZES, NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_f1_score = 0  # 初始化最高F1分数为0
    best_epoch = -1  # 初始化最佳epoch # 初始化迄今为止最高的F1分数为0

    for e in range(EPOCH):
        model.train()  # 确保模型处于训练模式
        for batch, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)  # 直接传入inputs
            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()

            # 假设这里有某种方式来计算y_pred和targets之间的F1分数
            # 实际代码中，可能需要在循环外部或使用不同的策略来计算F1分数

        dev_report = dev(model)

        f1_score = round(dev_report['macro avg']['f1-score'], 4)
        precision = round(dev_report['macro avg']['precision'], 4)  # 获取精确度值
        recall = round(dev_report['macro avg']['recall'], 4)  # 获取召回率值
        print(f'>> epoch: {e}, loss: {loss.item()}, dev_precision: {precision}, dev_recall: {recall}, dev_f1: {f1_score}')

        if f1_score > best_f1_score:
            best_f1_score = f1_score  # 更新最高F1分数
            best_epoch = e  # 更新最佳epoch
            # 保存当前模型的参数
            torch.save(model.state_dict(), MODEL_DIR + f'best_model.pth')


            # 最后，你可以打印出获得最高F1分数的epoch和分数
    print(f"Best F1 score of {best_f1_score} was achieved at epoch {best_epoch}")

