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


