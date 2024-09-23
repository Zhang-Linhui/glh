import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt

# 读取数据
# import pandas as pd
# X = pd.read_csv('data/X.csv', header=0).values
# adj_matrix = pd.read_csv('data/adjacency_matrix.csv', header=0).values
# y = pd.read_csv('data/Y.csv', header=0).values.squeeze()
# adj_matrix = np.array(adj_matrix, dtype=np.int64)
# # 对 X 和 y 进行类似的处理
# X = np.array(X, dtype=np.float32)
# y = np.array(y, dtype=np.int64)

x=np.load('data/features.npy')
y=np.load('data/labels.npy')
adj_matrix=np.load('data/adjacency_matrix.npy')

# 构建PyG的Data对象
edge_index = torch.LongTensor(adj_matrix.T)
x = torch.FloatTensor(x)
y = torch.LongTensor(y)
num_classes = len(torch.unique(y))
data = Data(x=x, edge_index=edge_index, y=y)
data.num_classes = num_classes
# 假设已经有了data对象
# 创建训练和测试掩码
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool) # type: ignore
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool) # type: ignore
# 示例：使用前半部分节点作为训练集，后半部分作为测试集
data.train_mask[:data.num_nodes // 2] = True  # type: ignore
data.test_mask[data.num_nodes // 2:] = True  # type: ignore
# 定义GCN模型


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 初始化模型
model = GCN(data.num_features, 16, data.num_classes)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
data.x = data.x.to(device)
data.edge_index = data.edge_index.to(device)
data.y = data.y.to(device)
# 如果有，也需要将train_mask, test_mask等移动到GPU
data.train_mask = data.train_mask.to(device)
data.test_mask = data.test_mask.to(device)


# 准备收集指标
train_losses = []
train_accuracies = []

# 训练模型
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    # 收集损失值
    train_losses.append(loss.item())
    
    # 计算训练准确率
    _, pred = out.max(dim=1)
    correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    acc = correct / int(data.train_mask.sum())
    train_accuracies.append(acc)
    
    # 每隔一定epoch输出一次指标
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {acc:.4f}')

# 评估模型
model.eval()
_, pred = model(data.x, data.edge_index).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print(f'Test Accuracy: {acc:.4f}')

# 绘制训练损失和准确率图表
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
