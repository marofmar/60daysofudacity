'''
Part8: Introduction to Plans
- Plan is an important object which is crucial to scale to industrial FL.
- Plan reduces bandwidth usage, allows asynchronous schemes and give more autonomy to remote devices.
- "Class of funcitons that you can transform into plans"
'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F
# the local worker should not be aa client worker. 
# Non client workers can store objects and we need this ability to run a plan. 
import syft as sy 
hook = sy.TorchHook(torch) 

# IMPORTANT: local worker should not be a client worker
hook.local_worker.is_client_worker = False 

server = hook.local_worker 

x11 = torch.tensor([-1,2.]).tag('input_data') 
x12 = torch.tensor([1,-2.]).tag('input_data2') 
x21 = torch.tensor([-1,2.]).tag('input_data') 
x22 = torch.tensor([1,-2.]).tag('input_data2')

device_1 = sy.VirtualWorker(hook, id = 'device_1', data = (x11, x12)) 
device_2 = sy.VirtualWorker(hook, id = 'device_2', data = (x21, x22)) 
devices = device_1, device_2 

#Basic example: defining a function that we want to transform into a plan. 
@sy.func2plan() 
def plan_double_abs(x):
	x = x+x 
	x = torch.abs(x) 
	return x 

plan_double_abs #<syft.msg.Plan.Plan at yadayada> 

'''
To use a plan, you need two things
- to build the plan
- to send it to a worker/device
'''

# Building a plan 
pointer_to_data = device_1.search('input_data')[0] 
plan_double_abs.is_built # False 
plan_double_abs.send(device_1) # fails 
plan_double_abs.build(torch.tensor([1., -2.])) # to build a plan, just call 'build' 

plan_double_abs.is_built #True 

plan_double_abs.send(device_1) # sucess 

pointer_to_result = plan_double_abs(pointer_to_data) 
print(pointer_to_result) 

pointer_to_result.get() # tensor([2., 4.]) 

#Towards a concrete example 
# apply Plan to Deep and Federated Learning 

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__() 
		self.fc1 = nn.Linear(2, 3) 
		self.fc2 = nn.Linear(3,2) 

	@sy.method2plan 
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim = 0) 

net = Net() 
net.forward 

#let's build the plan using some mock data 
net.forward.build(torch.tensor([1., 2.]))

net.send(device_1) 
pointer_to_data = device_1.search('input_data')[0] 
pointer_to_data = net(pointer_to_data)

pointer_to_result.get() 

# Switch between workers 
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__() 
		self.fc1 = nn.Linear(2,3) 
		self.fc2 = nn.Linear(3,2) 

	@sy.method2plan 
	def forward(self, x):
		x = F.relu(self.fc1(x)) 
		x = self.fc2(x) 
		return F.log_softmax(x, dim = 0) 

net = Net() 

# Build plan 
net.forward.build(torch.tensor([1., 2.]))

net.send(device_1) 
pointer_to_data = device_1.search('input_data')[0] 
pointer_to_result = net(pointer_to_dat) 
pointer_to_result.get() 

net.get() 

net.send(device_2) 
pointer_to_data = device_2.search('input_data')[0] 
pointer_to_result = net(pointer_to_data) 
pointer_to_result.get() 

# Automatically building plans that are functions 
# for functions(@sy.func2plan) we can automatically build the plan with no need to explicitly calling build.

@sy.func2plan(args_shape = [(-1,1)])
def plan_double_abs(x):
	x = x+x
	x = torch.abs(x) 
	return x

plan_double_abs.is_built #true.

@sy.func2plan(args_shape = [(1,2),(-1,2)])
def plan_sum_abs(x,y):
	s = x+y 
	return torch.abs(s) 

plan_sum_abs.is_built #True.











