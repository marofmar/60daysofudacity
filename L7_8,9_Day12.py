# assume syft installed

import torch as th 
import syft as sy 

hook = sy.TorchHook(th) 

bob = sy.VirtualWorker(hook, id = 'bob')
alice = sy.VirtualWorker(hook, id = 'alice')
jon = sy.VirtualWorker(hook, id = 'jon')

x = th.tensor([1,2,3,4,5]).sned(bob) # send x data to the worker bob

bob._object # shows what is inside objects in bob 

def extended_to_alice(x):
	x = x.send(alice)
	return x 

y = th.tensor([1,2,3,4,5]).send('bob').send('alice') # extened pointer! x->alice->bob

y= y.get() # now x->bob 

bob._objects # y inside 
alice._objets # {} empty inside! since y got back!

bob.clear_objects()
alice.clear_objects() # nothing left! Cleaning! 
# garbage collection 


