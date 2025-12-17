import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# 创建模型实例
model = SimpleNet()

# 检查是否有可用的 GPU，如果有就使用第一个 GPU，否则使用 CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 将模型移动到设备
model = model.to(device)

# 创建一些示例数据（这里使用随机生成的数据）
input_data = torch.randn(5, 10).to(device)
target = torch.randn(5, 1).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    output = model(input_data)

    # 计算损失
    loss = criterion(output, target)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印每个epoch的损失
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

print("Training finished!")
