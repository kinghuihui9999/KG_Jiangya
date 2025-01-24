from config import *
from utils import *
from model import *

# def dev(model):
#
#     id2label, _ = get_label()
#
#     test_dataset = Dataset('test')
#     test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#
#     loss_fn = nn.CrossEntropyLoss()
#
#     y_pred = []
#     y_true = []
#
#     with torch.no_grad():
#         for b, (input, mask, target) in enumerate(test_loader):
#
#             input = input.to(DEVICE)
#             mask = mask.to(DEVICE)
#             target = target.to(DEVICE)
#
#             test_pred = model(input, mask)
#             loss = loss_fn(test_pred, target)
#
#             # print('>> batch:', b, 'loss:', round(loss.item(), 5))
#
#             test_pred_ = torch.argmax(test_pred, dim=1)
#
#             y_pred += test_pred_.data.tolist()
#             y_true += target.data.tolist()
#
#     return evaluate(y_pred, y_true, id2label, output_dict=True)

def dev(model):
    id2label, _ = get_label()

    # 加载测试集
    test_dataset = Dataset(type='test', vocab= vocab, sample_path=TEST_SAMPLE_PATH)  # 确保传入正确的词汇表和测试集路径
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()

    y_pred = []
    y_true = []

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        for input, target in test_loader:
            input = input.to(DEVICE)
            target = target.to(DEVICE)

            output = model(input)  # 直接传入input，无需mask
            loss = loss_fn(output, target)

            pred = torch.argmax(output, dim=1)
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(target.cpu().numpy())

    # 评估模型性能
    return evaluate(y_pred, y_true, target_names=id2label, output_dict=True)

