import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from loss import ZINBLoss, MeanAct, DispAct

# create a transform to apply to each datapoint
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

	def __init__(self, input_dim=1422, hidden_dim=400, latent_dim=5,device=device):  # input dim is number of taxa
		super(VAE, self).__init__()

		# encoder
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.LeakyReLU(0.2)
		)

		# latent mean and variance
		#self.mean_layer = nn.Linear(latent_dim, 2)
		#self.logvar_layer = nn.Linear(latent_dim, 2)

		# decoder
		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, hidden_dim),
			nn.LeakyReLU(0.2),
			nn.Linear(hidden_dim, input_dim),
			nn.Sigmoid()
		)

		self._enc_mu = nn.Linear(hidden_dim, latent_dim)
		self._enc_logvar = nn.Linear(hidden_dim,latent_dim)
		self._dec_mean = nn.Sequential(nn.Linear(hidden_dim, input_dim), MeanAct())
		self._dec_disp = nn.Sequential(nn.Linear(hidden_dim, input_dim), DispAct())
		self._dec_pi = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Sigmoid())
		# loss function
		self.zinb_loss = ZINBLoss()

	def encode(self, x):
		x = self.encoder(x)
		mean, logvar = self._enc_mu(x), self._enc_logvar(x)
		return mean, logvar

	def reparameterization(self, mean, var):
		epsilon = torch.randn_like(var).to(device)
		z = mean + var * epsilon
		return z

	def decode(self, x):
		return self.decoder(x)

	def forward(self, x):
		latent_mean, logvar = self.encode(x) # get the mean and log variance of the latent space
		z = self.reparameterization(mean, logvar) # reparameterization trick
		x_hat = self.decode(z) # get the decoded output
		_mean = self._dec_mean(x_hat) # output of initial size
		_disp = self._dec_disp(x_hat) # dispersion estimate
		_pi = self._dec_pi(x_hat) # pi estimate
		return latent_mean, logvar, _mean, _disp, _pi

	def reconstruct(self, x):
		return self.decode(self.encode(data))



model = VAE().to(device)  # this switches the data/model/whatever to whatever device you want to work on (eg from CPU memory to a GPU)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam is an algorithm for stochastic optimization

def loss_function(mean, log_var):
	#reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
	KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
	return zinb + KLD

def KLD(mean, log_var):
	KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
	return KLD


def train(model, optimizer, epochs, device,features):
	model.train() # set model to training mode
	for epoch in range(epochs):
		overall_loss = 0
		for batch_idx, x in enumerate(train_loader):
			x = x.view(batch_size,features).to(device)

			optimizer.zero_grad()
			latent_mean, latent_logvar, mean, disp, pi = model(x)
			KLD_loss = KLD(latent_mean,latent_logvar)
			zinb_loss = model.zinb_loss(x,mean, disp, pi)
			loss = KLD_loss + zinb_loss
			overall_loss += loss.item()

			loss.backward()
			optimizer.step()

		print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / (batch_idx * batch_size))
	return overall_loss


output = train(model, optimizer, epochs=20, device=device,features=train_dataset.shape[1])

