# 首先导入必要的库和模块：
import os

from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torchvision.models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

# 定义常量
print('定义常量')
IMAGE_PATH = "/root/linux_deeplearning/dataset_forming"
DATA_PATH = "/root/linux_deeplearning/VM_dataset.csv"
IMAGE_SIZE = (640, 480)
BATCH_SIZE = 16

# 定义图像数据集类
print('定义图像数据集类')
class ImageDataset(Dataset):
    def __init__(self, image_path):
        self.image_path = image_path
        self.classes = os.listdir(self.image_path)
        self.num_classes = len(self.classes)
        self.image_files = []
        self.labels = []
        for i, cls in enumerate(self.classes):
            cls_path = os.path.join(self.image_path, cls)
            # 判断cls_path是否是文件夹
            if not os.path.isdir(cls_path):
                continue
            else:
                files = os.listdir(cls_path)

                self.image_files.extend([os.path.join(cls_path, f) for f in files])
                self.labels.extend([i] * len(files))
        self.transforms = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')  # 将图像转换为PIL Image类型
        if self.transforms:
            img = self.transforms(img)
        label = self.labels[idx]
        return img, label



# 定义离散数据集类
print('定义离散数据集类')
import pandas as pd
import numpy as np
import torch

class DiscreteDataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)
        self.data = self.data.dropna() # 删除具有缺失值的行
        self.features = torch.from_numpy(self.data.iloc[:, 1:-1].to_numpy(dtype=np.float32))
        self.labels = torch.from_numpy(self.data.iloc[:, -1].to_numpy(dtype=np.int32))
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.features[index], self.labels[index]






# 加载数据集
print('加载数据集')
image_dataset = ImageDataset(IMAGE_PATH)
discrete_dataset = DiscreteDataset(DATA_PATH)

# 创建数据加载器
print('创建数据加载器')
image_loader = DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=True)
discrete_loader = DataLoader(discrete_dataset, batch_size=BATCH_SIZE, shuffle=True)








# 我们需要训练基本模型,使用ResNet18模型训练:
# 定义ResNet18模型
print('定义ResNet18模型')
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.9.0',
                                     'resnet18',
                                     weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.resnet(x)

# 定义损失函数和优化器
print('定义损失函数和优化器')
num_classes = image_dataset.num_classes
resnet_model = ResNet18(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)


# 开始记录训练过程
writer = SummaryWriter()

# 定义训练函数，并记录到tensorboard
print('定义训练函数')
def train_resnet(model, criterion, optimizer, dataloader):
    model.train()
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss", loss.item())



# 开始训练ResNet18模型
print('开始训练ResNet18模型')
for epoch in range(1):

    train_resnet(resnet_model, criterion, optimizer, image_loader)
    print("Epoch:", epoch)



# 逻辑回归模型训练第二个数据集,使用sklearn库中的LogisticRegression类来训练逻辑回归模型，并使用准确度作为模型性能的评估指标
# 定义逻辑回归模型
logistic_model = LogisticRegression()

# 训练逻辑回归模型
features = discrete_dataset.features.numpy()
labels = discrete_dataset.labels.numpy()
logistic_model.fit(features, labels)

pred_labels = logistic_model.predict(features)
accuracy = accuracy_score(labels, pred_labels)
print("Accuracy on training set:", accuracy)
# 记录到tensorboard
writer.add_scalar("Accuracy", accuracy, global_step=0)




'''
我们已经得到了两个基本模型的训练结果。接下来，我们需要使用已训练的基本模型来提取特征，并将其输入到一个新的神经网络模型中。
我们将定义一个两个全连接层的神经网络模型，其中输入层接收两个基本模型的输出结果作为特征，输出层输出最终的分类结果。
'''
# 定义混合模型
class MixedModel(nn.Module):
    def __init__(self, num_classes, hidden_dim=64, dropout_prob=0.5):
        super(MixedModel, self).__init__()
        self.resnet = resnet_model.resnet
        self.logistic = nn.Linear(discrete_dataset.features.shape[1], num_classes)
        self.fc = nn.Sequential(
            nn.Linear(num_classes * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x1, x2):
        x1 = self.resnet(x1)
        x2 = self.logistic(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x





'''
定义损失函数和优化器，以及训练函数，用于训练混合模型。我们将使用交叉熵损失函数和Adam优化器进行模型训练。
'''
# 定义混合模型、损失函数和优化器
num_classes = image_dataset.num_classes
mixed_model = MixedModel(num_classes, hidden_dim=64, dropout_prob=0.5)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mixed_model.parameters(), lr=0.001)



def get_logistic_output(X, logistic_model):
    return torch.from_numpy(logistic_model.predict_proba(X)[:, 1]).unsqueeze(1)

def train_mixed(model, criterion, optimizer, image_loader, discrete_loader, logistic_model):
    model.train()
    for (image_inputs, image_labels), (discrete_inputs, discrete_labels) in zip(image_loader, discrete_loader):
        print(image_inputs.shape)  # 查看 image_inputs 的维度情况
        optimizer.zero_grad()
        # 将数据切片并合并样本批次和时间步维度
        image_inputs = image_inputs.permute(0, 2, 1, 3, 4)
        image_inputs = image_inputs.reshape((-1, 3, 640, 480))
        image_outputs = model.resnet(image_inputs)

        discrete_outputs = get_logistic_output(discrete_inputs, logistic_model)
        outputs = model(image_outputs, discrete_outputs)
        loss = criterion(outputs, image_labels)
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss", loss.item(), global_step=optimizer.state_dict()["param_groups"][0]["step"])


# 开始训练混合模型
for epoch in range(10):
    train_mixed(mixed_model, criterion, optimizer, image_loader, discrete_loader, logistic_model)
    print("Epoch:", epoch)

writer.close()

#
# '''
# 最后，我们可以使用训练好的混合模型对新的数据进行分类预测。我们可以使用图像数据集和离散数据集中的任意一种数据来进行预测，只需将数据输入到混合模型即可。
# '''
# # 使用混合模型进行分类预测
# def predict_mixed(model, image_input, discrete_input):
#     model.eval()
#     image_input = image_input.unsqueeze(0)
#     image_output = resnet_model(image_input)
#     discrete_output = logistic_model(discrete_input)
#     output = model(image_output, discrete_output)
#     pred_label = torch.argmax(output, dim=1)
#     return pred_label.item()
#
# # 加载一张图片和一个特征向量，用于分类预测
# test_image = cv2.imread("path/to/test/image")
# test_image = transforms.ToTensor()(test_image)
# test_discrete = torch.tensor([1, 2, 3, 4, 5])
#
# # 进行分类预测
# pred_label = predict_mixed(mixed_model, test_image, test_discrete)
# print("Predicted label:", pred_label)
