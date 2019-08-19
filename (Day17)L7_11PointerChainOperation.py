'''
Day 17: Tue 16 Jul 2019
1. Secure and Private AI Challenge
- Lesson7: 11. PointerChain Operations
	- shadow writing codes
	- try to get the concept of ‘move’ 
- Lesson8: 2. Project Demo Federated Learning with Trusted Aggregator 
	- shadow writing codes (course material) 
- Ask advices in tech_help channel in our SecurePrivateAI, Udacity, and fellow scholars were super kind and helpful, I am so thankful and grateful for them! 
'''

# L7 - 11

import torch as th 
#!pip install syft 
import syft as syft


hook = sy.TorchHook(th)
bob = sy.VirtualWorker(hook, id = 'bob')
alice = sy.VirtualWorker(hook, id = 'alice')

x = th.tensor([1,2,3,4,5]).send(bob).send(alice) 
bob._objects # so, x explicitly points the data
alice._objects # but the alice has some gadget pointing toward bob! (my understanding)

x.remote_get()
bob._objects  # so, if we 'remote_get()', then bob's objects go evaporated.
# empty objects in bob

alice._objects
# however, since bob's got evaporated, alice now has the 'object' which was belonged to bob before.
# and the object re-assigned to alice now has even has same id

x.move(bob)

print(x) # (Wrapper)>[PointerTensor | me:x -> bob:x]

# so now, alice and bob has the same obj but its id is diff
print(bob._objects)
print(alice._objects)


x = th.tensor([1,2,3,4,5]).send(bob)
bob._objects  # bob has [1,2,3,4,5]
alice._objects # alice does not have any,empty 
x.move(alice) # (Wrapper)>[PointerTensor | me:x-> alice:y] 

bob._objects # then bob is empty
alice._objects # now alice has what used to bob has before


# L8 - 02
import syft as sy 
import torch as th 
hook = sy.TorchHook(th) 
from torch import nn, optim 

#create a couple workers 

bob = sy.VirtualWorker(hook, id = 'bob')
alice = sy.VirtualWorker(hook, id = 'alice') 
secure_worker = sy.VirtualWorker(hook, id = 'secure_worker') 

bob.add_workers([alice, secure_worker])
alice.add_workers([bob, secure_worker]) 
secure_worekr.add_workers([alice, bob])
# hmm so all of them linked? 

# a tody dataset 
data = th.tensor([[0,0],[0,1],[1,0],[1,1.]], requires_grad = True) 
target = th.tensor([[0],[0],[1],[1.]], requires_grad = True)

# get pointers to training data on each worker by
# sending some training dat to bob and alice 
