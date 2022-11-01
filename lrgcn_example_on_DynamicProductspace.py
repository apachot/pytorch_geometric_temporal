try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import LRGCN

from torch_geometric_temporal.dataset import DynamicProductspaceDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
product_list = pd.read_csv('./HS6_labels.csv', header=0, index_col=False, sep=';')

loader = DynamicProductspaceDatasetLoader()

dataset = loader.get_dataset()

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = LRGCN(node_features, 16, 1, 1)
        self.linear = torch.nn.Linear(16, 1)

    def forward(self, x, edge_index, edge_weight, h_0, c_0):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h_0, c_0)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0
        
model = RecurrentGCN(node_features = 1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in tqdm(range(20)):
    cost = 0
    h, c = None, None
    for time, snapshot in enumerate(train_dataset):
        y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
        #print("snapshot.y", snapshot.y)
        cost_tmp = torch.mean((torch.abs(y_hat.T-snapshot.y))**2)
        #cost_tmp = torch.mean((torch.abs(y_hat.T-snapshot.y)))
        #print("y_hat",y_hat.T)
        #print("snapshot.y",snapshot.y)
        print("MSE_tmp: {:.10f}".format(cost_tmp))
        cost = cost + cost_tmp
        #cost = cost + torch.nn.MSELoss(y_hat.T, snapshot.y)
    cost = cost / (time+1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    
model.eval()
cost = 0
h, c = None, None
y_list, y_hat_list = [], []
for time, snapshot in enumerate(test_dataset):
    y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
    cost_tmp = torch.mean((torch.abs(y_hat.T-snapshot.y))**2)
    #cost_tmp = torch.mean((torch.abs(y_hat.T-snapshot.y)))
    cost = cost + cost_tmp
    # Store for analysis below
    y_list.append(snapshot.y.detach().cpu().numpy())
    y_hat_list.append(y_hat.T.detach().cpu().numpy())
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.10f}".format(cost))


Exports = []
line = pd.read_csv('hs_product_code.csv', header=0, index_col=0).to_numpy().tolist()
line = [str(i[0]).zfill(6) for i in line] 
Exports.append(line)
#pd.read_csv('hs_product_code.csv', header=0, index_col=0).values.tolist()

for time, snapshot in enumerate(dataset):
    line = snapshot.x.detach().numpy().tolist()
    line = [i[0] for i in line]
    #print("ligne", len(line))
    Exports.append(line)
    #exporting node weights for gephi
    gephi_df = product_list
    gephi_df['weight'] = line
    print("gephi_df",gephi_df)
    pd.DataFrame(gephi_df).to_csv('./gephi/predict_exports_year_'+str(time+1995)+'.csv', header=True, index=False)
# Prediction for 2021
last_year = dataset[-1]
h, c = None, None
y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
line = y_hat.detach().numpy().tolist()
line = [i[0] for i in line]
#line = [str(i[0]).zfill(6) for i in line]
#exporting node weights for gephi
gephi_df = product_list
gephi_df['weight'] = line
print("gephi_df",gephi_df)
pd.DataFrame(gephi_df).to_csv('./gephi/predict_exports_forecast.csv', header=True, index=False)

Exports.append(line)

pd.DataFrame(Exports).to_csv('predict_exports.csv', header=False, index=False)
print("y_hat_list", y_hat_list)
print("y_list", y_list)


'''
product = 1



preds = np.asarray([pred[0][product] for pred in y_hat_list])
print("preds", preds)
labs  = np.asarray([label[product] for label in y_list])
print("Data points:,", preds.shape)

plt.figure(figsize=(20,5))

#plt.plot(hs_product_code)
sns.lineplot(data=preds, label="pred")
sns.lineplot(data=labs, label="true")
plt.show()
'''
'''
year = 10

print("y_hat_list", y_hat_list[0][0])
print("y_list", y_list[0])


plt.figure(figsize=(20,5))

#plt.plot(hs_product_code)
sns.set_theme(style="whitegrid")
labels = [i.zfill(6) for i in product_list[1:-1][1].values]
print("labels", len(labels))
data = pd.DataFrame(y_list[0], labels, ["Actuals"])
data['Predictions'] = y_hat_list[0][0]


sns.lineplot(data=data, linewidth=1, markers=False, dashes=True)
plt.xticks(fontsize=8, rotation=45)
plt.show()
'''

