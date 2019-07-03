'''
Federated learning
- A technique for training Machine Learning models on data to which you do not have access.

PySyft

'''
import torch as th
x = th.tensor([1,2,3,4,5])
print(x)
y = x+x
print(y)

import syft as sy
hook = sy.TorchHook(th)
th.tensor([1,2,3,4,5])
