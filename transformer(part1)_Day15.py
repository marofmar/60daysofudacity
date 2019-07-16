"""
Day 15: Sat 13 Jul 2019
1. Secure and Private AI Challenge
- 18:00 local meetup in Seoul!!!!!!!: real fun time, also productive! I could ask questions about my personal project, and got sufficient answers from each of the members who are smart, nice, and kind people! It was real good time with them =) 
- Joined study group: DataSyfters
- @Tyler Yang shared great source about PyTorch and Transformer (http://nlp.seas.harvard.edu/2018/04/03/attention.html) 

"""

import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math, copy, time 
from torch.autograd import Variable 
import seaborn 
seaborn.set_context(context = 'talk') # ?
%matplotlib inline 


class EncoderDecoder(nn.Module):
	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		super(EncoderDecoder, self).__init__() 
		self.encoder = encoder 
		self.decoder = decoder 
		self.src_embed = src_embed 
		self.tgt_embed = tgt_embed
		self.generator = generator 

	def forward(self, src, tgt, src_mask, tgt_mask):
		"Take in and process masekd src anad target sequences." 
		return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask) 

	def encode(sefl, src, src_mask):
		return self.encoder(self.src_embed(src), src_mask) 

	def decode(self, memory, src_maask, tgt, tgt_mask):
		return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
	"Define standard linear + softmax generation step."
	def __init__(self, d_model, vocab):
		super(Generator, self).__init__() 
		self.proj = nn.Linear(d_model, vocab) 

	def forward(self, x):
		return F.log_softmax(self.proj(x), dim = -1)
