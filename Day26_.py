'''
Further step from Day25_.py
- send model to correct worker
- train on the daata located there 
- get the model back and repeat the next worker 

'''

import syft as sy 
hook = sy.TorchHook(torch) 

# create a couple workers 
bob = sy.VirtualWorker(hook, id = "bob")
alice = sy.VirtualWorker(hook, id = "alice")

# a toy dataset 
data = torch.tensor([[0,0],[0,1.],[1.,0],[1,1.]], requires_grad = True)
target = torch.tensor([[0],[0],[1.],[1.]], requires_grad = True) 

# get pointers to training data on each worekr by
# sending soe training data to bob and alice 
data_bob = data[0:2]
target_bob = target[0:2]

data_alice = data[2:]
target_alice = target[2:]

# initialize a toy model 
model = nn.Linear(2,1) 

data_bob = data_bob.send(bob) 
data_alice = data_alice.send(alice) 
target_bob = target_bob.send(bob)
target_alice = target_alice.send(alice) 

# organize pointers into a list 
datasets = [(data_bob, target_bob), (data_alice, target_alice)]

#opt = optim.SGD(params = model.parameters(), lr = 0.1) 

def train():
	# training logic 
	opt = optim.SGD(parameters = model.parameters(), lr = 0.1) 
	for iter in range(10):

		#new) iterate through each worekr's dataset 
		for data, target in datasets:

			# NEW) send model to correct worker 
			model.send(data.location) 
			# 1 erase previous gradients (if they exist) 
			opt.zero_grad() 
			# 2 make a prediction 
			pred = model(data) 
			# 3 calculate how much missed
			loss = ((pred-target)**2).sum() 
			# 4 figure out which weights cuased missings
			loss.backward() 
			# 5 change those weights 
			opt.step() 
			# NEW) get model (with gradients)
			model.get() 
			# 6 print out progress
			print(loss.get()) 

train()













