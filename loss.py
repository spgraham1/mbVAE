# code from https://github.com/ttgump/scDCC/blob/master/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ZINBLoss(nn.Module):
	def __init__(self):
		super(ZINBLoss, self).__init__()

	def forward(self, x, mean, disp, pi, log_var, scale_factor=1.0, ridge_lambda=0.0, beta=1.0):
		#first calculate the ZINB loss
		eps = 1e-10
		scale_factor = scale_factor[:, None]
		mean = mean * scale_factor

		#log likelihood
		t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
		t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
		nb_final = t1 + t2

		nb_case = nb_final - torch.log(1.0 - pi + eps) #adjust nb_final to account for zeros (pi)
		zero_nb = torch.pow(disp / (disp + mean + eps), disp)
		zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
		result = torch.where(torch.le(x, 1e-8), zero_case, nb_case) #this combines ZINB and NB -- uses ZINB at elements where the value of X < 1e-8

		if ridge_lambda > 0:
			ridge = ridge_lambda * torch.square(pi)
			result += ridge

		result = torch.mean(result)
		return result


class GaussianNoise(nn.Module):
	def __init__(self, sigma=0):
		super(GaussianNoise, self).__init__()
		self.sigma = sigma

	def forward(self, x):
		if self.training:
			x = x + self.sigma * torch.randn_like(x)
		return x


class MeanAct(nn.Module):
	def __init__(self):
		super(MeanAct, self).__init__()

	def forward(self, x):
		return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
	def __init__(self):
		super(DispAct, self).__init__()

	def forward(self, x):
		return torch.clamp(F.softplus(x), min=1e-4, max=1e4)