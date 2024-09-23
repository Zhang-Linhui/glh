import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tensorboardX import SummaryWriter

# 加载数据
x = np.load('data/features.npy')
y = np.load('data/labels.npy')
adj_matrix = np.load('data/adjacency_matrix.npy')

# 构建PyG的Data对象
edge_index = torch.LongTensor(adj_matrix.T)
x = torch.FloatTensor(x)
y = torch.LongTensor(y)
num_classes = len(torch.unique(y))
data = Data(x=x, edge_index=edge_index, y=y)
data.num_classes = num_classes

# 创建训练和测试掩码
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool) # type: ignore
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool) # type: ignore
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
data.train_mask = data.train_mask.to(device)
data.test_mask = data.test_mask.to(device)

# 创建 SummaryWriter 实例
writer = SummaryWriter('runs/experiment_1')

# 训练循环
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # 记录训练损失
    writer.add_scalar('Loss/train', loss.item(), epoch)

    # 记录训练准确率
    _, pred = out[data.train_mask].max(dim=1)
    correct = (pred == data.y[data.train_mask]).sum().item()
    acc = correct / data.train_mask.sum().item()
    writer.add_scalar('Accuracy/train', acc, epoch)

    print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {acc}')

# 关闭 writer
writer.close()
