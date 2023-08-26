import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()

        # 空间平均池化层，将特征图转换为1D向量
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层，用于计算特征图每个通道的注意力权重
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # 空间平均池化，将特征图转换为1D向量
        y = self.avg_pool(x).view(b, c)

        # 通过全连接层计算权重，将1D向量映射到[0, 1]之间
        weights = self.fc(y).view(b, c, 1, 1)

        # 将权重扩展到与输入特征图相同的形状
        weights = weights.expand_as(x)

        # 将权重与输入特征图相乘，得到加权的特征图
        out = x * weights

        return out

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiTaskModel, self).__init__()

        # 下载并加载预训练的DenseNet权重
        densenet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)

        # 获取DenseNet的特征提取部分
        self.base_model = densenet.features

        num_features = densenet.classifier.in_features

        # 添加注意力机制模块
        self.attention = AttentionModule(num_features)

        self.fc_main = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self.fc_aux1 = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.fc_aux2 = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        shared_features = self.base_model(x)

        # 应用注意力机制模块
        shared_features = self.attention(shared_features)

        shared_features = F.adaptive_avg_pool2d(shared_features, (1, 1))
        shared_features = shared_features.view(shared_features.size(0), -1)

        main_output = self.fc_main(shared_features)
        aux1_output = self.fc_aux1(shared_features)
        aux2_output = self.fc_aux2(shared_features)

        return main_output, aux1_output, aux2_output


# 定义网络结构
# num_classes = 4  # AD分类任务类别数

# model = MultiTaskModel(num_classes)

# 将输入数据传递给网络进行前向计算
# input_data = torch.randn(1, 3, 224, 224)  # 示例输入数据
# main_output, aux1_output, aux2_output = model(input_data)

# 打印输出结果
# print('Main task output:', main_output)
# print('Auxiliary task 1 output (Hippocampus):', aux1_output)
# print('Auxiliary task 2 output (Cortical Brain Structure):', aux2_output)