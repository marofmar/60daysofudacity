'''
Part12: Train an Encrypted NN on Encrypted Data 
- both model and data encrypted
'''

# Step1: Create workers and toy data 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim. 
import syft as sy 

# Setup
hook = sy.TorckHook(torch) 
alice = sy.VirtualWorker(id = 'alice', hook = hook)
bob = sy.VirtualWorker(id = 'bob', hook = hook) 
james = sy.VirtualWorker(id = 'james', hook = hook) 

# Toy Dataset 
data = torch.tensor([[0,0],[0,1],[1,0],[1,1.]])
target = torch.tensor([[0],[0],[1],[1.]]) 

# A Toy Model 
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__() 
		self.fc1 = nn.Linear(2,2) 
		self.fc2 = nn.Linear(2,1) 

	def forward(self, x):
		x = self.fc(x) 
		x = F.relu(x) 
		x = self.fc2(x)
		return x 
model = Net() 

# Step2: Encrypt the model and data 
'''
Sincee SMPC(Secure Multi-Party Computation) only works on integers, 
to operate over numbers with decimal points (weights, activation, etc), 
in need of encoding all numbers using Fixed Precision
'''

# encode everything 
data = data.fix_precision().share(bob, alice, crypto_provider = james, requires_grad = True) 
target = target.fix_precision().share(bob, alice, crypto_provider = james,requires_grad = True)
model = model.fix_precision().share(bob, alice, crypto_provider = james, requires_grad = True) 

print(data) # 

# Step 3: Train 
opt = optim.SGD(params = model.parameters(), lr = 0.1).fix_precision() 

for iter in range(20):
	# 1. erase previous gradients if they exist 
	opt.zero_grad() 
	# 2. make a prediction 
	pred = model(data) 
	# 3. calculate how much we missed 
	loss = ((pred-target)**2).sum() 
	# 4. figure out which weights should be blamed 
	loss.backward() 
	# 5. change the weights, update
	opt.step() 
	# 6. print our process 
	print(loss.get().float_precision()) 

	








