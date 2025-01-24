from config import *
from utils import *
from model import *

if __name__ == '__main__':

    id2label, _ = get_label()

    test_dataset = Dataset('test')
    test_loader = data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    model = TextCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR + 'model_5.pth', map_location=DEVICE))
    loss_fn = nn.CrossEntropyLoss()

    y_pred = []
    y_true = []

    with torch.no_grad():
        for b, (input, mask, target) in enumerate(test_loader):

            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            test_pred = model(input, mask)
            loss = loss_fn(test_pred, target)

            print('>> batch:', b, 'loss:', round(loss.item(), 5))

            test_pred_ = torch.argmax(test_pred, dim=1)

            y_pred += test_pred_.data.tolist()
            y_true += target.data.tolist()

    print(evaluate(y_pred, y_true, id2label))

# from config import *
# from utils import *
# from model import TextCNN
#
# if __name__ == '__main__':
#     id2label, label2id = get_label()
#
#     # 加载测试数据集
#     test_dataset = Dataset(type='test', vocab=vocab, sample_path=TEST_SAMPLE_PATH)  # 确保使用正确的路径和词汇表
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#
#     # 初始化模型
#     model = TextCNN(VOCAB_SIZE, EMBEDDING_DIM, NUM_FILTERS, FILTER_SIZES, NUM_CLASSES).to(DEVICE)
#     model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'MODELWORD2VEC.pth'), map_location=DEVICE))
#
#     loss_fn = nn.CrossEntropyLoss()
#
#     y_pred = []
#     y_true = []
#
#     model.eval()  # 设置模型为评估模式
#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             inputs = inputs.to(DEVICE)
#             targets = targets.to(DEVICE)
#
#             outputs = model(inputs)  # 直接传入inputs，无需mask
#             loss = loss_fn(outputs, targets)
#
#             preds = torch.argmax(outputs, dim=1)
#             y_pred.extend(preds.cpu().numpy())
#             y_true.extend(targets.cpu().numpy())
#
#     # 使用实际的评估函数
#     report = evaluate(y_pred, y_true, target_names=[id2label[i] for i in range(len(id2label))])
#
#     print(report)
