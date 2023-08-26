import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from torchvision.transforms import transforms

# 定义数据集类
class BrainDataset(Dataset):
    def __init__(self, Dataset):
        self.data_dir = Dataset
        self.image_files = []

        # 获取所有图像文件的路径
        for root, dirs, files in os.walk(Dataset):
            print('root:',root)
            print('dirs:', dirs)
            print('files:', files)
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    self.image_files.append(os.path.join(root, file))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    def __getitem__(self, index):
        image_path = self.image_files[index]
        # print('image_path:', image_path)
        label = os.path.basename(os.path.dirname(image_path))

        # 读取图像并应用变换
        image = Image.open(image_path)
        image = self.transform(image)

        return image, label

    def __len__(self):
        print('len',len(self.image_files))
        return len(self.image_files)