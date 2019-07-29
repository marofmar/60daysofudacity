'''
Day27
Part3: Advanced Remote Execution Tools 
call send(): to send model to the location of training data
call get(): brining the trained result after updating it 

Now, we will average out the gradients before calling .get() 
so that we won't see anyone's exact gradient
'''

import torch 
import syft as sy 
hook = sy.TorchHook(torch) 

#3.1 pointers to pointers 
bob = sy.VirtualWorker(hook, id = 'bob')
alice = sy.VirtualWorker(hook, id = 'alice') 

x = torch.tensor([1,2,3,4]) # local tensor 
x_ptr = x.send(bob) # send the local tensor to bob, x_ptr is a pointer 

pointer_to_x_ptr = x_ptr.send(alice) #send the pointer to alice
# this is not MOVEing, as bob still has the tensor 
bob._objects #still bob has actual data 
alice._objects #what alice has is a pointer, pointing to bob from alice  'PointerTensor'

x_ptr = pointer_to_x_ptr.get()  #to get x_ptr back from alice 
x = x_ptr.get() # to get x back from bob 

#3.1.2 Arithmetic on Pointer->Pointer->Data Object 
bob._objects #empty {}
alice._objects #empty {} 

p2p2x = torch.tensor([1,2,3,4,5]).send(bob).send(alice) 
y = p2p2x + p2p2x 

bob._objects #has two tensors
alice._objects #has two pointers pointing to bob from alice 

y.get().get() # returns 2,4,6,8,10 tensor 
bob._objects # only has one data left 
alice._objects # only has one pointer left 

p2p2x.get().get() 
bob._objects #empty {}
alice._objects #empty {} 











