import torch
import torchvision
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingLR


labels_frame = pd.read_csv('./train.csv')
leaves_labels = sorted(list(set(labels_frame['label'])))

num_classes = len(leaves_labels)
class2num = dict(zip(leaves_labels, range(num_classes)))
num2class = {v: k for k, v in class2num.items()}


class LeavesDataset(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, resize_height=224, resize_width=224):
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.file_path = file_path
        self.mode = mode

        self.data_info = pd.read_csv(csv_path, header=None)
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == 'train':
            self.train_image = np.asarray(self.data_info.iloc[1:self.train_len, 0])
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len, 1])
            self.image_arr = self.train_image
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image

        self.real_len = len(self.image_arr)

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'.format(mode, self.real_len))

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(self.file_path + single_image_name)

        if self.mode == 'train':
            train_augs = torchvision.transforms.Compose([
                # 随机裁剪图像，所得图像为原始面积的0.08到1之间，高宽比在3/4和4/3之间。
                # 然后，缩放图像以创建224 x 224的新图像
                torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
                torchvision.transforms.RandomHorizontalFlip(),
                # 随机更改亮度，对比度和饱和度
                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                # 添加随机噪声
                torchvision.transforms.ToTensor(),
                # 标准化图像的每个通道
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

        else:
            valid_test_augs = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                # 从图像中心裁切224x224大小的图片
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

        if self.mode == 'train':
            img_as_img = train_augs(img_as_img)
        else:
            img_as_img = valid_test_augs(img_as_img)

        if self.mode == 'test':
            return img_as_img
        else:
            label = self.label_arr[index]
            number_label = class2num[label]
            return img_as_img, number_label

    def __len__(self):
        return self.real_len


train_path = './train.csv'
test_path = './test.csv'
image_path = './'  # csv文件中已经images的路径了，因此这里只到上一级目录

train_dataset = LeavesDataset(train_path, image_path, 'train')
valid_dataset = LeavesDataset(train_path, image_path, 'valid')
test_dataset = LeavesDataset(test_path, image_path, 'test')

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=5
)

valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=5
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=5
)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


device = get_device()


def set_parameter_requires_grad(model, feature_extracting):
    """ 模型冻结 """
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False


def res_model(num_classes, feature_extracting=False, use_pretrained=True):
    model_ft = torchvision.models.resnet34(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extracting)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft


# 超参数
# learning_rate = 1e-4
# weight_decay = 3e-3
# num_epoch = 20
# model_path = 'models/classify_leaves_v1.pth'
#
# # 初始化模型
# model = res_model(num_classes)
# model = model.to(device)
# model.device = device

# 损失函数 - 交叉熵
# loss = nn.CrossEntropyLoss()
#
# # 优化器
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler = CosineAnnealingLR(optimizer, T_max=10)
#
# # 迭代次数
# n_epochs = num_epoch
#
# best_acc = 0.0
# if __name__ == '__main__':
#     for epoch in range(n_epochs):
#         # ----------------- Train -----------------
#         model.train()
#         train_loss = []
#         train_accs = []
#         for batch in tqdm(train_loader):
#             imgs, labels = batch
#             imgs = imgs.to(device)
#             labels = labels.to(device)
#             logits = model(imgs)
#             l = loss(logits, labels)
#
#             optimizer.zero_grad()
#             l.backward()
#             optimizer.step()
#             acc = (logits.argmax(dim=-1) == labels).float().mean()
#
#             train_loss.append(l.item())
#             train_accs.append(acc)
#
#         train_loss = sum(train_loss) / len(train_loss)
#         train_acc = sum(train_accs) / len(train_accs)
#         print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
#         scheduler.step()
#
#         # ----------------- Validation -----------------
#         model.eval()
#         valid_loss = []
#         valid_accs = []
#
#         for batch in tqdm(valid_loader):
#             imgs, labels = batch
#             with torch.no_grad():
#                 logits = model(imgs.to(device))
#
#             l_v = loss(logits, labels.to(device))
#             acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
#
#             valid_loss.append(l_v.item())
#             valid_accs.append(acc)
#
#         valid_loss = sum(valid_loss) / len(valid_loss)
#         valid_acc = sum(valid_accs) / len(valid_accs)
#
#         print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
#
#         if valid_acc > best_acc:
#             best_acc = valid_acc
#             torch.save(model.state_dict(), model_path)
#             print('saving model with acc {:.3f}'.format(best_acc))


# predict
if __name__ == '__main__':
    saveFileName = './submission.csv'
    model_path = './models/classify_leaves_v0.pth'

    model = res_model(176)

    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    prediction = []
    for batch in tqdm(test_loader):
        imgs = batch
        with torch.no_grad():
            logits = model(imgs.to(device))
        prediction.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    preds = []
    for i in prediction:
        preds.append(num2class[i])

    test_data = pd.read_csv(test_path)
    test_data['label'] = pd.Series(preds)
    submission = pd.concat([test_data['image'], test_data['label']], axis=1)
    submission.to_csv(saveFileName, index=False)
    print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
