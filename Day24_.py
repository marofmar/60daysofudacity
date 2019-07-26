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

# 1. 
x = torch.tensor([1,2,3,4,5])
y = x+ x

# 1.1
'''
- to perform computations on some other machines
- Thus, instead of using Torch tensors, gonnaa work with pointers to tensors.
'''
bob = sy.VirtualWorker(hook, id = 'bob') 
x = torch.tensor([1,2,3,4,5])
y = torch.tensor([1,1,1,1,1]) 

x_ptr = x.send(bob)
y_ptr = y.send(bob) # send the tensors to Bob 

print(x_ptr, bob._objects) 

z = x_ptr + x_ptr 
bob._objects # now three including z 

'''
x_ptr.locaation: bob
x_ptr.id_at_loaction: random integer id where the tensor stored
x_ptr.id: random integer, the id of our pointer
x_ptr.owner: me, the worker which owns the pointer tensor 
'''
x_ptr.locaation 
bob 
bob == x_ptr.location # true 
x_ptr.id_at_loaction 
x_ptr.owner 

me = sy.local_worker 
me # 'me' worker automatically created when we call "hook = sy.TorchHook()"

print(me == x_ptr.owner) # True 

x_ptr.get() # get back from "send" 

y_ptr
y_ptr.get() 
z.get() 
bob._objects # nothing left in bob {} 








