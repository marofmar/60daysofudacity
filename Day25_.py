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