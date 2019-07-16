

import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math, copy, time 
from torch.autograd import Variable 
import matplotlib.pyplot as plt 
import seaborn 
seaborn.set_context(context='talk') 
%matplotlib.inline 

class EncoderDecoder(nn.Module):
	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder 
		self.decoder = decoder
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed 
		self.generator = generator

	def forward(self, src, tgt, src_mask, tgt_mask):
		return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

	def encode(self, src, src_mask):
		return self.encoder(self.src_embed(src), src_mask) 

	def decode(self, memory, src_mask, tgt, tgt_mask):
		return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask) 

class Generator(nn.Module):
	def __init__(self, d_model, vocab):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, vocab) 

	def forward(self, x):
		return F.log_softmax(self.proj(x), dim = -1) 


#The encoder is composed of aa stack of N = 6 identical layers 
def clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
	def __init__(self, layer, N):
		super(Encoder, self).__init__() 
		self.layers = clones(layer, N) 
		self.norm = LayerNorm(layer.size) 

	def forward(self, x, mask):
		for layer in self.layers:
			x = laeyr(x, maask) 

		return self.norm(x) 

class LayerNorm(nn.Module):
	def __init__(self, features, eps = 1e-6):
		super(LayerNorm, self).__init__() 
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features)) 
		self.eps = eps 

	def forward(self, x):
		mean = x.mean(-1, keepdim = True) 
		std = x.std(-1, keepdim = True) 
		return self.a_2*(x-mean)/(std+self.eps)+self.b_2










