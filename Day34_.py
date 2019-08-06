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
				100. * baatch_idx / len(train_loader), loss.item())) 

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








