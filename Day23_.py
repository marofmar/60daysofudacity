# Concept 8 review
import torch as th 
#!pip install syft 
import syft as syft


hook = sy.TorchHook(th)
bob = sy.VirtualWorker(hook, id = 'bob')
alice = sy.VirtualWorker(hook, id = 'alice')
secure_worker = sy.VirtualWorker(hook, id = 'secure_worker')

x = th.tensor([1,2,3,4,5]) 
x = x.share(bob, alice, secure_worker) # shared

y = x+x 
bob._objects 
y.get() #decrpyts the encrypted tensors and return [2,4,6,8,10]


x = th.tensor([0.1,0.2,0.3,0.4,0.5])
x = x.fix_prec() 
x = x.float_prec() 

type(x.child.child) # ohhhh child child...?!

type(x.child) # child?!!!


y = y.get().gloat_prec() 