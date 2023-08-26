import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import evaluate
# 引入模型
# from VGG_classifier import VGG
from MultiTaskModel import MultiTaskModel

from brain_dataset import BrainDataset

# 设置文件路径和参数
data_dir = 'D:\OASIS\select_png\All'  # 数据集路径
data_dir_GM = 'D:\OASIS\select_png\GM'
data_dir_Hip = 'D:\OASIS\select_png\Hip'
batch_size = 16  # 批量大小
num_epochs = 2 # 训练的轮数
learning_rate = 0.0001  # 学习率
class_num = 2
# 将打印输出保存到文件
# sys.stdout = open('D:\实验相关记录\ori_0.6_0.4\\training_output.txt', 'w')
# 定义数据集类

# 定义多任务分类网络模型

# 测试模型
test_dataset = BrainDataset('D:\OASIS\\test_png')
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 创建数据集实例和数据加载器
Alldataset = BrainDataset(data_dir)

GMdataset = BrainDataset(data_dir_GM)

Hipdataset = BrainDataset(data_dir_Hip)

# print('dataset:', dataset)
Alldataloader = DataLoader(Alldataset, batch_size=batch_size, shuffle=True)

GMdataloader = DataLoader(GMdataset, batch_size=batch_size, shuffle=True)

Hipdataloader = DataLoader(Hipdataset, batch_size=batch_size, shuffle=True)
# 创建模型实例和损失函数
model = MultiTaskModel(class_num)

# 创建带逻辑的二元交叉熵损失函数实例
criterion = nn.BCEWithLogitsLoss()

# 定义优化器(Adam,L2正则化)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_correct = 0
    total_samples = 0
    epoch_predictions = []
    epoch_labels = []
    total_loss = 0  # 总体损失

    for (images1, labels1), (images2, labels2), (images3, labels3) in zip(Alldataloader, GMdataloader, Hipdataloader):

        # 梯度清零
        optimizer.zero_grad()

        # 正向传播
        main_outputs, _, _ = model(images1)
        print('main_outputs:', main_outputs)
        _, hippocampus_aux_outputs, _ = model(images2)
        # print('hippocampus_aux_outputs:', hippocampus_aux_outputs)
        _, _, GM_aux_outputs = model(images3)
        # print('GM_aux_outputs:', GM_aux_outputs)
        # 将标签转换为独热编码
        class_labels = ['HC', 'AD']

        onehot_labels1 = torch.zeros(len(labels1), len(class_labels))
        for i, label in enumerate(labels1):
            class_index = class_labels.index(label)
            onehot_labels1[i][class_index] = 1

        onehot_labels2 = torch.zeros(len(labels2), len(class_labels))
        for i, label in enumerate(labels2):
            class_index = class_labels.index(label)
            onehot_labels2[i][class_index] = 1

        onehot_labels3 = torch.zeros(len(labels3), len(class_labels))
        for i, label in enumerate(labels3):
            class_index = class_labels.index(label)
            onehot_labels3[i][class_index] = 1

        # 计算主分类任务损失
        main_loss = criterion(main_outputs, onehot_labels1)
        print('main_loss:', main_loss)

        # 计算辅助任务损失(使用Huber损失函数)
        hippocampus_aux_loss = criterion(hippocampus_aux_outputs, onehot_labels2)
        print('hippocampus_aux_loss:', hippocampus_aux_loss)
        GM_aux_loss = criterion(GM_aux_outputs, onehot_labels3)
        # print('GM_aux_loss:', GM_aux_loss)
        # 总体损失为主分类任务损失加上辅助任务损失
        total_loss = main_loss + 0.8 * hippocampus_aux_loss + 0.2 * GM_aux_loss  # 超参数加权求和
        # print('total_loss:', total_loss)
        # 反向传播和优化

        total_loss.backward()  # 通过反向传播计算出模型参数的梯度
        optimizer.step()  # 更新模型的参数
        # 计算分类准确度
        # print('main_outputs.data:', main_outputs.data)
        _, predicted_labels = torch.max(main_outputs.data, 1)
        print("a:", predicted_labels)
        labels = labels1
        print('b:', labels)
        # 预定义的标签到编码的映射
        label_map = {'HC': 0, 'AD': 1}
        # 将标签转换为数字编码
        numeric_labels = [label_map[label] for label in labels]
        # 将数字编码的标签转换为Tensor类型
        labels = torch.tensor(numeric_labels)

        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)

        # 保存预测和标签用于后续计算指标
        # epoch_predictions.extend(predicted_labels.cpu().numpy())
        # epoch_labels.extend(labels.cpu().numpy())
    # 打印训练信息
    # print(f'Epoch [{epoch + 1}/{num_epochs}], Total Loss: {total_loss.item()}')
    # 计算分类准确度、Precision、Recall、F1-Score、混淆矩阵
    # accuracy = total_correct / total_samples
    # print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy * 100}%')
    # report = classification_report(epoch_labels, epoch_predictions, target_names=['CN', 'MCI'])
    # print(f'Epoch [{epoch + 1}/{num_epochs}], report:\n{report}')
    # cm = confusion_matrix(epoch_labels, epoch_predictions)
    # print('confusion_matrix:\n', cm)
    print(f'Epoch [{epoch + 1}/{num_epochs}]')
    evaluate.evaluate_model(model, test_dataloader)
# 测试模型
print('ori_1.0_0.7')


