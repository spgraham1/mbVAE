import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

# create a transofrm to apply to each datapoint
transform = transforms.Compose([transforms.ToTensor()])

#read the training data
train_dataset = pd.read_csv('~/Desktop/mbVAE/training_data_100k.tsv',sep='\t')
train_dataset = train_dataset.to_numpy()
train_dataset = torch.tensor(train_dataset,dtype=torch.float32)
test_dataset = pd.read_csv('~/Desktop/mbVAE/test_data.tsv',sep='\t')
test_dataset = test_dataset.to_numpy()
test_dataset = torch.tensor(test_dataset,dtype=torch.float32)

#alt + shift + e to run selected code block

# create train and test dataloaders
batch_size = 100
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):

	def __init__(self, input_dim=1422, hidden_dim=400, latent_dim=200,device=device):  # input dim is number of taxa
		super(VAE, self).__init__()

		# encoder
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.LeakyReLU(0.2),
			nn.Linear(hidden_dim, latent_dim),
			nn.LeakyReLU(0.2)
		)

		# latent mean and variance
		self.mean_layer = nn.Linear(latent_dim, 2)
		self.logvar_layer = nn.Linear(latent_dim, 2)

		# decoder
		self.decoder = nn.Sequential(
			nn.Linear(2, latent_dim),
			nn.LeakyReLU(0.2),
			nn.Linear(latent_dim, hidden_dim),
			nn.LeakyReLU(0.2),
			nn.Linear(hidden_dim, input_dim),
			nn.Sigmoid()
		)

	def encode(self, x):
		x = self.encoder(x)
		mean, logvar = self.mean_layer(x), self.logvar_layer(x)
		return mean, logvar

	def reparameterization(self, mean, var):
		epsilon = torch.randn_like(var).to(device)
		z = mean + var * epsilon
		return z

	def decode(self, x):
		return self.decoder(x)

	def forward(self, x):
		mean, logvar = self.encode(x)
		z = self.reparameterization(mean, logvar)
		x_hat = self.decode(z)
		return x_hat, mean, logvar

model = VAE().to(device)  # this switches the data/model/whatever to whatever device you want to work on (eg from CPU memory to a GPU)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam is an algorithm for stochastic optimization

def loss_function(x, x_hat, mean, log_var):
	reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
	#reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
	KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
	return reproduction_loss + KLD

def train(model, optimizer, epochs, device,features):
	model.train()
	for epoch in range(epochs):
		overall_loss = 0
		for batch_idx, x in enumerate(train_loader): #the _ is for the 'target data', which we don't have (it's basically a place holder)
			x = x.view(batch_size,features).to(device)

			optimizer.zero_grad()

			x_hat, mean, log_var = model(x)
			loss = loss_function(x, x_hat, mean, log_var)

			overall_loss += loss.item()

			loss.backward()
			optimizer.step()

		print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / (batch_idx * batch_size))
	return overall_loss


output = train(model, optimizer, epochs=50, device=device,features=train_dataset.shape[1])

