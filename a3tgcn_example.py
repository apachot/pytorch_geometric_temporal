try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN

from torch_geometric_temporal.dataset import ProductspaceDatasetLoader

from torch_geometric_temporal.signal import temporal_signal_split

loader = ProductspaceDatasetLoader()

dataset = loader.get_dataset()

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(RecurrentGCN, self).__init__()
        self.recurrent = A3TGCN(node_features, 100, periods)
        self.linear = torch.nn.Linear(100, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x.view(x.shape[0], 1, x.shape[1]), edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
        
model = RecurrentGCN(node_features = 1, periods = 2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in tqdm(range(50)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr).T
        #print("y_hat", y_hat)
        #print("snapshot.y", snapshot.y)
        #print("y_hat-snapshot.y", (y_hat-snapshot.y))
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    
model.eval()
cost = 0
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr).T
    #print("y_hat", y_hat)
    #print("snapshot.y", snapshot.y)
    #print("y_hat-snapshot.y", (y_hat-snapshot.y))
    #print("torch.mean((y_hat-snapshot.y)**2)", torch.mean((y_hat-snapshot.y)**2))
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))

Exports = []
line = pd.read_csv('hs_product_code.csv', header=0, index_col=0).to_numpy().tolist()
line = [str(i[0]).zfill(6) for i in line] 
Exports.append(line)
#pd.read_csv('hs_product_code.csv', header=0, index_col=0).values.tolist()

for time, snapshot in enumerate(dataset):
    line = snapshot.y.detach().numpy().tolist()
    #print("ligne", len(line))
    Exports.append(line)
# Prediction for 2021
last_year = dataset[-1]
y_hat = model(last_year.x, last_year.edge_index, last_year.edge_attr)
line = y_hat.detach().numpy().tolist()
line = [str(i[0]).zfill(6) for i in line] 
Exports.append(line)

pd.DataFrame(Exports).to_csv('predict_exports_FRA.csv', header=False, index=False)