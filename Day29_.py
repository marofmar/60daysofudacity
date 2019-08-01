"""
Part5: Sandbox
https://github.com/OpenMined/PySyft/blob/dev/examples/tutorials/Part%2005%20-%20Welcome%20to%20the%20Sandbox.ipynb

"""

import torch 
import syft as sy 
sy.create_sandbox(globals()) 

'''
Content of Sandbox
- Hooking PyTorch
- Creating virtual worekrs: bob, theo, jason, alice, andy, jon 
- Storing hook and workers as global variables 
- Loading datasets from Scikit Learn: Boston housing, Diabetes dataset, Breast Cancer Dataset
- Digits Datasets: Iris Dataset, Wine Dataset, Linnerud Dataset 
- Distributing daatasets among workers 
- Collecting workers into a virtualGrid 
'''

workers # six virtual workers listed
hook # TorchHook
bob # one of the virtual workers

'''
Worker Search Functionality
- how we can search for datasets on a remote mahine
eg. a researcher(me) want to queary hospital data, not leaving the lab
'''

x = torch.tensor([1,2,3,4,5.]).tag("#fun", "#boston", "#housing").describe("The input datapoints to the boston housing dataset.")
y = torch.tensor([1,2,3,4,5.]).tag("#fun", "#boston", "#housing").describe("The input datapoints to the boston housing dataset.")
z = torch.tensor([1,2,3,4,5.]).tag("#fun", "#mnist",).describe("The image in the MNIST training dataset.") 

x
# tensor 1,2,3,4,5 printed out
# Tags printed out
# Description printed out
# Shape of the tensor which is [5] printed out.

x = x.send(bob) 
y = y.send(bob) 
z = z.send(bob) 

results = bob.search("#boston", "#housing") #this searches for exact matchd within a tag or within the description 
results #four tensors listed up

print(results[0].description) # recall the description of the Boston Housing Dataset 

'''
Virtual Grid 
- A Grid is a simply a collection of workers which gives you some convenient functions for when you want to put together a dataset
# hmm, pretty new for me
'''

grid = sy.VirtualGrid(*workers) 
results, tag_ctr = grid.search("#boston") 
# 4 results found in bob, and 2 from each other 5 worekr

boston_target, _ = grid.search("#boston", "#target") 






