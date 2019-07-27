'''
Pysyft Tutorial
Section 1.2 Using Tensor Pointers
'''
# 0. Setup
import syft as sy 
from syft.fraameworks.torch.pointers import PointerTensor 
from syft.fraameworks.torch.tensors.decorators import LogginTensor 
import sys 
import torch 

hook = sy.TorchHook(torch) 
from torch.nn import Parameter 
import torch.nn as nn 
import torch.nn.functional as F 

bob = sy.VirtualWorker(hook, id = 'bob') 
x = torch.tensor([1,2,3,4,5]).send(bob)
y = torch.tensor([1,1,1,1,1]).send(bob) 

z = x + y  
'''
instead of computing an addition locally, a command was serialized aand sent to Bob, which performed the computation,
created a tensor z, and then returned the pointer to z back.
'''
z.get()  # get on the pointer, we receive the result back to our machine.

z = torch.add(x, y) # also works 
z.get() # same result 

'''
Variables (including backpropagation) 
'''

x = torch.tensor([1,2,3,4,5], requires_grad = True).send(bob) 
y = torch.tensor([1,1,1,1,1], requires_grad = True).send(bob) 

z = (x+y).sum() 
z.backward() 
x = x.get() 

'''
Part 2: Intro to Federated Learning
In Federated Learning, instead of bringing all the trainable data into central server,
we bring the model to the trainable data! 
'''

'''
Section 2.1 - A Toy Federated Learning Example 
- a toy dataset 
- a model 
- some basic training logic

'''

import torch 
from torch import nn 
from torch import optim 

# a tody dataset 
data = torch.tensor([[0,0],[0,1],[1,0].[1,1.1]], requires_grad = True)
target = torch.tensor([[0],[0],[1],[1.]], requires_grad = True) 

# a toy model 
model = nn.Linear(2,1) 

def train():
	# training logic 
	opt = optim.SGD(params = model.parameters(), lr = 0.1) 
	for iter in range(20):
		# 1 erase previous gradients (if they exist) 
		opt.zero_grad() 
		# 2 make a prediction 
		pred = model(data) 
		# 3 calculate how much we missed 
		loss = ((pred-target)**2).sum() 
		# 4 figure out which weights caused us to miss 
		loss.backward() 
		# 5 change those weights
		opt.step() # wow, so the step was the process of updating paraameters! 
		# 6 print our progress 
		print(loss.data) 












