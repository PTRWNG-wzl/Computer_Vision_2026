#!/usr/bin/env python3
""" 按步骤运行 coursework_02 - 使用较少迭代以便快速完成 """
import os
os.chdir('/Users/zhilinwang/Documents/GitHub/Computer_Vision_2026/coursework_02')

print("=" * 60)
print("步骤 1: 导入库")
print("=" * 60)
import tarfile
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import time
import random
import matplotlib
matplotlib.use('Agg')  # 无头模式，不弹窗
import matplotlib.pyplot as plt
from matplotlib import colors

print("导入完成。")

print("\n" + "=" * 60)
print("步骤 2: 数据已解压 (Task01_BrainTumour_2D/)")
print("=" * 60)
assert os.path.exists('Task01_BrainTumour_2D/training_images'), "请先下载并解压数据集"
print("数据集存在。")

print("\n" + "=" * 60)
print("步骤 3: Q1 可视化 4 张训练图像")
print("=" * 60)
train_img_path = 'Task01_BrainTumour_2D/training_images'
train_label_path = 'Task01_BrainTumour_2D/training_labels'
image_names = sorted(os.listdir(train_img_path))
indices = random.sample(range(len(image_names)), 4)
fig, axes = plt.subplots(4, 2, figsize=(6, 12))
seg_cmap = colors.ListedColormap(['black', 'green', 'blue', 'red'])
for i, idx in enumerate(indices):
    image = imageio.v2.imread(os.path.join(train_img_path, image_names[idx]))
    label = imageio.v2.imread(os.path.join(train_label_path, image_names[idx]))
    axes[i, 0].imshow(image, cmap='gray')
    axes[i, 0].set_title('Image')
    axes[i, 0].axis('off')
    axes[i, 1].imshow(label, cmap=seg_cmap, vmin=0, vmax=3)
    axes[i, 1].set_title('Label map')
    axes[i, 1].axis('off')
plt.tight_layout()
plt.savefig('step3_visualisation.png', dpi=100)
plt.close()
print("已保存 step3_visualisation.png")

print("\n" + "=" * 60)
print("步骤 4: Q2 数据集类 (BrainImageSet)")
print("=" * 60)
def normalise_intensity(image, thres_roi=1.0):
    val_l = np.percentile(image, thres_roi)
    roi = (image >= val_l)
    mu, sigma = np.mean(image[roi]), np.std(image[roi])
    eps = 1e-6
    return (image - mu) / (sigma + eps)

class BrainImageSet(Dataset):
    def __init__(self, image_path, label_path='', deploy=False):
        self.image_path = image_path
        self.deploy = deploy
        self.images = []
        self.labels = []
        image_names = sorted(os.listdir(image_path))
        for image_name in image_names:
            image = imageio.v2.imread(os.path.join(image_path, image_name))
            self.images += [image]
            if not self.deploy:
                label = imageio.v2.imread(os.path.join(label_path, image_name))
                self.labels += [label]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = normalise_intensity(self.images[idx])
        label = self.labels[idx]
        return image, label

    def get_random_batch(self, batch_size):
        indices = [random.randint(0, len(self) - 1) for _ in range(batch_size)]
        images, labels = [], []
        for idx in indices:
            image, label = self[idx]
            images.append(np.expand_dims(image, axis=0))
            labels.append(label)
        return np.stack(images, axis=0).astype(np.float32), np.stack(labels, axis=0).astype(np.int64)

train_set = BrainImageSet('Task01_BrainTumour_2D/training_images', 'Task01_BrainTumour_2D/training_labels')
test_set = BrainImageSet('Task01_BrainTumour_2D/test_images', 'Task01_BrainTumour_2D/test_labels')
print(f"训练集: {len(train_set)} 张, 测试集: {len(test_set)} 张")

print("\n" + "=" * 60)
print("步骤 5: Q3 U-Net 模型")
print("=" * 60)
class UNet(nn.Module):
    def __init__(self, input_channel=1, output_channel=1, num_filter=16):
        super(UNet, self).__init__()
        n = num_filter
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, n, kernel_size=3, padding=1),
            nn.BatchNorm2d(n), nn.ReLU(),
            nn.Conv2d(n, n, kernel_size=3, padding=1),
            nn.BatchNorm2d(n), nn.ReLU()
        )
        n *= 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(n/2), n, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n), nn.ReLU(),
            nn.Conv2d(n, n, kernel_size=3, padding=1),
            nn.BatchNorm2d(n), nn.ReLU()
        )
        n *= 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(n/2), n, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n), nn.ReLU(),
            nn.Conv2d(n, n, kernel_size=3, padding=1),
            nn.BatchNorm2d(n), nn.ReLU()
        )
        n *= 2
        self.conv4 = nn.Sequential(
            nn.Conv2d(int(n/2), n, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n), nn.ReLU(),
            nn.Conv2d(n, n, kernel_size=3, padding=1),
            nn.BatchNorm2d(n), nn.ReLU()
        )
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        self.upconv3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU()
        )
        self.conv_out = nn.Conv2d(16, output_channel, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        conv1_skip = x
        x = self.conv2(x)
        conv2_skip = x
        x = self.conv3(x)
        conv3_skip = x
        x = self.conv4(x)
        x = self.upconv1(x)
        x = torch.cat([x, conv3_skip], dim=1)
        x = self.conv5(x)
        x = self.upconv2(x)
        x = torch.cat([x, conv2_skip], dim=1)
        x = self.conv6(x)
        x = self.upconv3(x)
        x = torch.cat([x, conv1_skip], dim=1)
        x = self.conv7(x)
        return self.conv_out(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
model = UNet(input_channel=1, output_channel=4, num_filter=16).to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

print("\n" + "=" * 60)
print("步骤 6: Q4 训练 (使用 500 迭代快速验证)")
print("=" * 60)
model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
num_iter = 500  # 快速验证用
train_batch_size = 16
eval_batch_size = 16

start = time.time()
for it in range(1, 1 + num_iter):
    model.train()
    images, labels = train_set.get_random_batch(train_batch_size)
    images = torch.from_numpy(images).to(device, dtype=torch.float32)
    labels = torch.from_numpy(labels).to(device, dtype=torch.long)
    logits = model(images)
    optimizer.zero_grad()
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    if it % 100 == 0:
        print(f'  Iter {it}: train loss = {loss.item():.4f}')
    if it % 250 == 0:
        model.eval()
        with torch.no_grad():
            timg, tlab = test_set.get_random_batch(eval_batch_size)
            timg = torch.from_numpy(timg).to(device, dtype=torch.float32)
            tlab = torch.from_numpy(tlab).to(device, dtype=torch.long)
            tloss = criterion(model(timg), tlab)
        print(f'  Iter {it}: test loss = {tloss.item():.4f}')

torch.save(model.state_dict(), os.path.join(model_dir, 'model_500.pt'))
print(f'训练完成，耗时 {time.time()-start:.1f}s，模型已保存')

print("\n" + "=" * 60)
print("步骤 7: Q5 部署与可视化")
print("=" * 60)
model.load_state_dict(torch.load(os.path.join(model_dir, 'model_500.pt'), map_location=device))
model.eval()
test_indices = random.sample(range(len(test_set)), 4)
seg_cmap = colors.ListedColormap(['black', 'green', 'blue', 'red'])
fig, axes = plt.subplots(4, 3, figsize=(9, 12))
with torch.no_grad():
    for i, idx in enumerate(test_indices):
        image, label_gt = test_set[idx]
        img_t = torch.from_numpy(np.expand_dims(np.expand_dims(image, 0), 0)).to(device, dtype=torch.float32)
        logits = model(img_t)
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title('Test image')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(pred, cmap=seg_cmap, vmin=0, vmax=3)
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(label_gt, cmap=seg_cmap, vmin=0, vmax=3)
        axes[i, 2].set_title('Ground truth')
        axes[i, 2].axis('off')
plt.tight_layout()
plt.savefig('step7_deployment.png', dpi=100)
plt.close()
print("已保存 step7_deployment.png")

print("\n" + "=" * 60)
print("全部步骤完成!")
print("=" * 60)
print("输出文件: step3_visualisation.png, step7_deployment.png, saved_models/model_500.pt")
