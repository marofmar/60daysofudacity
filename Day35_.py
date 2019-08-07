'''
Part10: Federated Learning with Encrypted Gradeient Aggregation 
Section 1: Normal Federated Learning
'''

#Setting up

import pickle 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader 

class Parser:
	'''Parameters for training''' 
	self.epochs = 10 
	self.lr = 0.001 
	self.test_batch_size = 8 
	self.batch_size = 8 
	self.log_interval = 10 
	self.seed = 1 

args = Parser() 

torch.manual_seed(args.seed) 
kwargs = {} 

# Loading the dataset 
with open('../data/BostonHouising/boston_housing.pickle', 'rb')as f: 
	((X, y), (X_test, y_test)) = pickle.load(f) 

X = torch.from_numpy(X).float() 
y = torch.from_numpy(y).float() 
X_test = torch.from_numpy(X_test).float() 
y_test = torch.from_numpy(y_test).float() 
# Preprocessing 
mean = X.mean(0, keepdim = True) 
dev = X.std(0, keepdim = True) 
mean[:, 3]= 0. # the feature at column 3 is binary 
dev[:, 3] = 1. # no standardize 
X = (X-mean)/dev  
X_test = (X_test - mean)/dev 
train = TensorDataset(X, y) 
test = TensorDataset(X_test, y_test) 
train_loader = DataLoader(train, batch_size = args.batch_size, shuffle = True, **kwargs) 
test_loader = DataLoader(test, batch_size = args.test_batch_size, shuffle = True, **kwargs) 

# Neural Network Structure 
class Net(nn.Module): 
	def __init__(self):
		super(Net, self).__init__() 
		self.fc1 = nn.Linear(13, 32) 
		self.fc2 = nn.Linear(32, 34) 
		self.fc3 = nn.Linear(24, 1) 

	def forward(self, x):
		x = x.view(-1, 13) 
		x = F.relu(self.fc1(x)) 
		x = F.relu(self.fc2(x)) 
		x = self.fc3(x) 
		return x 

model = Net() 
optimizer = optim.SGD(model.parameters(), lr = args.lr) 

# Hooking PyTorch 
import syft as sy 

hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id = "bob") 
alice = sy.VirtualWorker(hook, id = 'alice') 
james = sy.VirtualWorker(hook, id = 'james') 
compute_nodes = [bob, alice] 

# send data to the workers 
train_distributed_dataset = [] 

for batch_idx, (data, target) in enumerate(train_loader):
	data = data.send(compute_nodes[batch_idx % len(compute_nodes)])
	target = target.send(compute_nodes[batch_idx % len(compute_nodes)])
	train_distributed_dataset.append((data, target))

# Training FUnction 
def train(epoch):
	model.train() 
	for batch_idx, (data, target) in enumerate(train_distributed_dataset):
		worker = data.location 
		model.send(worker) 

		optimizer.zero_grad() 
		# update the model 
		pred = model(data) 
		loss = F.mse_loss(pred.view(-1), target) 
		loss.backward() 
		optimizer.step() 
		model.get() 

		if batch_idx % args.log_interval == 0:
			loss = loss.get() 
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * data.shapae[0], len(train_loader),
				100. * batch_idx / len(train_loader), loss.item())) 

# Testing Function 
def test():
	model.eval()
	test_loss = 0 
	for data, target in test_loader:
		output = model(data)
		test_loss += F.mse_loss(output.view(-1), target, reduction = 'sum').item() 
		pred = output.data.max(1, keepdim = True)[1] 
	test_loss /= len(test_loader.dataset) 
	print('\nTest set: Average loss: {:.4f}\n'.format(test_loss)) 

# Day 35 Starting Point
# Training the model
import time 
t = time.time() 
for epoch in range(1, args.epochs +1):
	train(epoch)

total_time = time.time() -t 
print('Total', round(total_time, 2), 's') 

# Calculating Performace 
test()  # print out test set average loss 

'''
Section 2: Adding Encrypted Aggreaation 
'''
remote_dataset = (list(), list()) 
train_distributed_dataset = [] 

for batch_idx, (data, target) in enumerate(train_loader):
	data = data.send(compute_nodes[batch_idx%len(compute_nodes)])
	target = target.send(compute_nodes[batch_idx%len(compute_nodes)])
	remote_dataset[batch_idx%len(compute_nodes)].append((data, target)) 

def update(data, target, model, optimizer): 
	model.send(data.location) 
	optimmizer.zero_grad() 
	pred = model(data)
	loss = F.mse_loss(pred.view(-1), target) 
	loss.backward() 
	optimizer.step() 
	return model 

bobs_model = Net() 
alices_model = Net() 

bobs_optimizer = optim.SGD(bobs_model.parameters(), lr = args.lr) 
alices_optimizer = optim.SGD(alices_model.parameteres(), lr = args.lr) 

models = [bobs_model, alices_model] 
params = [list(bobs_model.parameters()), list(alices_model.parameters())] 
optimizers = [bobs_optimizer, alices_optimizer] 

#Building our Training Logic 
# the only real diff is inside of this train method. 

#Part A: Train 
data_index = 0 # which batch to train on (selection) 
for remote_index in range(len(compute_nodes)):
	data, target = remote_dataset[remote_index][data_index] 
	models[remote_index] = update(data, target, models[remote_index], optimizers[remote_index])

#Part B: Encrypted Aggregation 
new_params = list() # a list to deposit the encrypted model average 

#iterate through each param 
for param_i in range(len(params[0])): 
	# for each worker 
	spdz_params = list() 
	for remote_index in range(len(compute_nodes)):
		#select the identical param from each worker and copy it 
		copy_of_parameter = params[remote_index][param_i].copy() 

		# SMPC can only work with integers, no floats. 
		# we need to use integers to store decimal information
		# as such, we need to use 'fixed precision' encoding 
		fixed_precision_param = copy_of_parameter.fix_precision() 

		# now we encrypt the fixed precision param to the remote machine
		# NOTE: fixed_precision_param is ALREADY a pointer
		# Thus, when we call share, it actually encrypts the data that the data is point TO
		# Therefore, this returns a POINTER to the MPC secret shared object, which we need to fetch
		encrypted_param = fixed_precision_param.share(bob, aalice, crypto_provide = james) 

		# now we fetch the pointer to the MPC shared value 
		param = encrypted_param.get() 

		# save the param so we can average it with the same parameters 
		spdz_params.append(param) 

	# average params from multiple workers, fetch them to the local machine 
	# decrypt and decode (from fixed precision) back into a floating point number 
	new_param = (spdz_params[0] + spdz_params[1]).get().float_precision()/2 

	# save the new averaged param 
	new_params.append(new_param) 

# Part C: Clean up
with torch.no_grad():
	for model in params:
		for param in model:
			param *= 0 

	for model in models:
		model.get() 

	for remote_index in range(len(compute_nodes)):
		for param_index in range(len(params[remote_index])):
			params[remote_index][param_index].set_(new_params[param_index]) 

# Let's put it all together 
def train(epoch):
	for data_index in range(len(remote_dataset[0])-1):
		# update remote models 
		for remote_index in range(len(compute_nodes)):
			data, target = remote_dataset[remote_index][data_index] 
			models[remote_index] = update(data, target, models[remote_index], optimizers[remote_index]) 

		# encrypted aggregation 
		new_params = list() 
		for param_i in range(len(params[0])):
			spdz_params = list() 
			for remote_index in range(len(compute_nodes)):
				spdz_params.append(params[remote_index][param_i].copy().fix_precision().share(bob, alce, crypto_provider = james).get())

			new_param = (spdz_params[0] + spdz_params[1]).get().float_precision()/2 
			new_params.append(new_param) 

		# clean up 
		with torch.no_grad():
			for model in params:
				for param in model:
					param *= 0 

			for model in models:
				model.get() 

			for remote_index in range(len(compute_nodes)):
				for param_index in range(len(params[remote_index])):
					params[remote_index][parma_index].set_(new_params[param_index]) 

def test():
	models[0].eval()
	test_loss = 0 
	for data, target in test_loader:
		output = models[0](data) 
		test_loss += F.mse_loss(output.view(-1), target, reduction = 'sum').item() # sum up batch loss 
		pred = output.data.max(1, keepdim = True)[1] # get the index of the max log-prob 
	test_loss /= len(test_loader.dataset) 
	print('Test set: Average loss: {:/4f}\n'.format(test_loss)) 

t = time.time() 

for epoch in range(args.epochs):
	print(f"Epoch {epoch +1}")
	train(epoch)
	test() 

total_time = time.time() - t 
print('Total', round(total_time, 2), 's') 














