import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import anndata as ad
import scanpy as sc
#import torchvision.transforms as transforms
#from torchvision.utils import save_image, make_grid
from loss import ZINBLoss, MeanAct, DispAct
from sklearn.preprocessing import StandardScaler

# create a transform to apply to each datapoint
#transform = transforms.Compose([transforms.ToTensor()])

#read the training data
train_dataset = pd.read_csv('~/Desktop/mbVAE/training_data_100k.tsv',sep='\t')
train_dataset = train_dataset.to_numpy()
train_dataset = torch.tensor(train_dataset,dtype=torch.float32)
test_dataset = pd.read_csv('~/Desktop/mbVAE/test_data.tsv',sep='\t')
test_dataset = test_dataset.to_numpy()
test_dataset = torch.tensor(test_dataset,dtype=torch.float32)

train_dataset = pd.read_csv('~/Desktop/mbVAE/raw_balanced_samples.csv',sep=',')
sample_counts = train_dataset.sum(axis=1)
size_factor = sample_counts / np.median(sample_counts)
raw = train_dataset.to_numpy() # get the raw training data
#norm = ad.AnnData(raw) # copy the raw dataset to normalize
#sc.pp.normalize_total(norm) # get the normalized dataset
scaler = StandardScaler()
norm = scaler.fit_transform(train_dataset)
# convert all to tensors
raw_train_dataset = torch.tensor(raw,dtype=torch.float32)
norm_train_dataset = torch.tensor(norm,dtype=torch.float32)
size_factor = torch.tensor(size_factor,dtype=torch.float32)

combined_training_set = TensorDataset(norm_train_dataset, raw_train_dataset, size_factor)

#alt + shift + e to run selected code block

# create train and test dataloaders
batch_size = 100
train_loader = DataLoader(dataset=combined_training_set, batch_size=batch_size, shuffle=True)
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
		z = self.reparameterization(latent_mean, logvar) # reparameterization trick
		x_hat = self.decode(z) # get the decoded output
		_mean = self._dec_mean(x_hat) # output of initial size
		_disp = self._dec_disp(x_hat) # dispersion estimate
		_pi = self._dec_pi(x_hat) # pi estimate
		return latent_mean, logvar, _mean, _disp, _pi

	def reconstruct(self, x):
		return self.decode(self.encode(x))



model = VAE().to(device)  # this switches the data/model/whatever to whatever device you want to work on (eg from CPU memory to a GPU)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam is an algorithm for stochastic optimization

def loss_function(mean, log_var):
	#reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
	KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
	return KLD

def KLD(mean, log_var):
	KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
	return KLD


def train(model, optimizer, epochs, device,features):
	model.train() # set model to training mode
	for epoch in range(epochs):
		overall_loss = 0
		#for batch_idx, x in enumerate(train_loader):
		for batch_idx, (norm_data, raw_data, batch_sf) in enumerate(train_loader):
			# x = x.view(batch_size,features).to(device)
			# x_norm, x_raw, batch_sf = x.view(batch_size,features).to(device)
			x_norm = norm_data.view(batch_size,features).to(device)
			x_raw = raw_data.view(batch_size,features).to(device)
			#batch_sf = batch_sf.view(batch_size,features).to(device)
			optimizer.zero_grad()
			latent_mean, latent_logvar, mean, disp, pi = model(x_norm)
			KLD_loss = KLD(latent_mean,latent_logvar)
			zinb_loss = model.zinb_loss(x_raw,mean, disp, pi)
			loss = KLD_loss + zinb_loss
			overall_loss += loss.item()

			loss.backward()
			optimizer.step()

		print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / (batch_idx * batch_size))
	return overall_loss


output = train(model, optimizer, epochs=5, device=device,features=train_dataset.shape[1])

latent_mean, latent_logvar, mean, disp, pi = model(train_dataset)

meanNP = mean2.detach().numpy()
meanNP = pd.DataFrame(meanNP)
meanNP.to_csv('~/Desktop/mbVAE/zinb_mean_output_abrv.csv')

inputNP = input2.detach().numpy()
inputNP = pd.DataFrame(inputNP)
inputNP.to_csv('~/Desktop/mbVAE/zinb_input_abrv.csv')

t_np = t.numpy() #convert to Numpy array


df = pd.DataFrame(t_np) #convert to a dataframe
df.to_csv("testfile",index=False) #save to file
