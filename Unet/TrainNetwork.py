import torch
from torch import nn, optim
from torchvision import transforms
import ISBI_dataloader as dl
import Unet_Structure


# 训练函数
def train_runner(model, device, dataloader, optimizer, epoch):
    # 输出值
    loss_sum = 0.0
    item_num = 0
    # 设置损失函数
    lossfunction = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6]).to(device), )  # 给边界像素赋予更高权重
    for index, data in enumerate(dataloader, 1):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 初始化梯度
        optimizer.zero_grad()
        # 计算损失
        outputs = model(inputs)
        outputs = transforms.Resize((572, 572))(outputs)
        # labels=torch.squeeze(labels).long()
        loss = lossfunction(outputs, labels)
        loss_sum += loss
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        print(f"eopch:{epoch},loss_avg:{loss_sum / index}")


# 加载数据集
dataset = dl.ISBI_dataloader("data/train/image", "data/train/label")
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=3, shuffle=False)
# 部署gpu
device = torch.device("cuda:0")
# 加载网络
model = Unet_Structure.UNet(1, 1).to(device)
# 定义优化器
learning_rate = 0.0002
momentum = 0.9
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
# 开始训练
for epoch in range(1, 20):
    train_runner(model, device, loader, optimizer, epoch)
