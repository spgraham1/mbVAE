import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


#data = pd.read_csv('~/Desktop/mbVAE/PRJNA318004_disease_counts.tsv',sep='\t')
data = pd.read_csv('~/Desktop/mbVAE/PRJNA318004_disease_countsNorm.tsv',sep='\t')
meta = pd.read_csv('~/Desktop/mbVAE/PRJNA318004_disease_meta.tsv',sep='\t')
metaNP = meta['status'].values
dataNP = data.to_numpy()
class CRC_data:
	def __init__(self, data, meta):
		self.data = torch.from_numpy(data)
		self.meta = torch.from_numpy(meta)

	def __len__(self):
		return(len(self.data))
	def __getitem__(self,idx):
		return self.data[idx],self.meta[idx]

data_train,data_test,meta_train,meta_test = train_test_split(dataNP,metaNP,test_size=0.2)

class MLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(1422, 200),
			nn.LeakyReLU(),
			nn.Linear(200, 100),
			nn.LeakyReLU(),
			nn.Linear(100, 1)
		)
	def forward(self, x):
		return self.layers(x)


dataset = CRC_data(data_train,meta_train)
testset = CRC_data(data_test,meta_test)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)

mlp = MLP()

loss_function = nn.L1Loss()
optimizer = torch.optim.Adagrad(mlp.parameters(),lr=1e-4)

for epoch in range(0,5):
    print(f'Starting Epoch {epoch+1}')
    current_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))
        optimizer.zero_grad()
        outputs = mlp(inputs)
        lossF = loss_function(outputs, targets)
        lossF.backward()
        optimizer.step()
        current_loss += lossF.item()
        if i%10 == 0:
            print(f'Loss after mini-batch %5d: %.3f'%(i+1, current_loss/500))
            current_loss = 0.0
    print(f'Epoch {epoch+1} finished')
print("Training has completed")


test_data = torch.from_numpy(data_test).float()
test_targets = torch.from_numpy(meta_test).float()

with torch.no_grad():
    outputs = mlp(test_data)
    predicted_labels = outputs.squeeze().tolist()

predicted_labels = np.array(predicted_labels)
test_targets = np.array(test_targets)

mse = mean_squared_error(test_targets, predicted_labels)
r2 = r2_score(test_targets, predicted_labels)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
