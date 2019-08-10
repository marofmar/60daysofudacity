'''
Part 12 bis- Encrypted Training on MNIST 
Part X - Secure Training anad Evaluation on MNIST 

Machine Learning as a Service solution: MLaaS 

'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms 
import time 

class Arguments():
	def __init__(self):
		self.batch_size = 64 
		self.test_batch_size = 64 
		self.epochs = 10 
		self.lr = 0.02 
		self.seed = 1 
		self.log_interval = 1 
		self.precision_fractional = 3 

args = Arguments() 

_ = torch.manual_seed(args.seed) 

import syft as sy 
hook = sy.TorchHook(torch) 

def connect_to_workers(n_workers):
	return [
		sy.VirtualWorker(hook, id = f'worker{i+1}')
		for i in range(n_workers) 
	]

def connect_to_crypto_provider():
	return sy.VirtualWorker(hook, id = 'crypto_provider') 

workers = connect_to_workers(n_workers = 2) 
crypto_provider = connect_to_crypto_provider() 

# Getting access and secret share data 

# only part of dataset will be used for efficiency purpose
n_train_items = 640 
n_test_items = 640 

def get_private_data_loader(precision_functional, workers, crypto_provider):

	def one_hot_of(index_tensor):
		onehot_tensor = torch.zeros(*index_tensor.shape, 10) # 10 classees of MNIST 
		onehot_tensor = onehot_tensor.scatter(1, index_tensor.view(-1,1),1) 
		return onehot_tensor 

	def secret_share(tensor):
		return (
			tensor.fix_precision(precision_functional = precision_functional).share(*workers, crypto_provider = crypto_provider, requires_grad = True)

		)

	transformation = transforms.Copose([
		transfomrs.ToTensor(),
		transforms.Normalize((0.1307,),(0.3081,))
	])

	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train= True, download = True, transform=transformation),
		batch_size = args.batch_size
	)


	private_train_loader = [
		(secret_share(data), secret_share(one_hot_of(target))) 
		for i, (data, target) in enumerate(train_loader)
		if i <n_train_items / args.batch_size 
	]

	test_loader = torch.utils.data.DataLoaer(
		datasets.MNIST('../data', train = True, download = True, transform = transformation),
		batch_size = args.test_batch_size 
	)

	private_test_loader = [
		(secret_share(data), secrete_share(target.float())) 
		for i, (data, target) in enumerate(test_loader) 
		if i < n_test_items / args.test_batch_size 
	]


	return private_train_loader, private_test_loader 

private_train_loader, private_test_loader = get_private_data_loader(
	precision_fractional= args.precision_fractional,
	workers = workers,
	crypto_provider = crypto_provider) 

# Model Specification 
class Net(nn.Module): 
	def __init__(self):
		super(Net, self).__init__() 
		self.fc1 = nn.Linear(28*28, 128) 
		self.fc2 = nn.Linear(128, 64) 
		self.fc3 = nn.Linear(64, 10) 

	def forward(self, x):
		x = x.view(-1, 28*28) 
		x = F.relu(self.fc1(x)) 
		x = F.relu(self.fc2(x)) 
		x = self.fc3(x) 
		return x 

# Trainin and testing functions 
def train(args, model, private_train_loader, optimizer, epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(private_train_loader):
		start_time = time.time() 
		optimizer.zero_grad() 
		output = model(data) 

		batch_size = output.shape[0] 
		loss = ((output-target)**2).sum() 

		loss.backward() 

		optimizer.step() 

		if batch_idx % args.log_interval == 0:
			loss = loss.get().float_precision() 
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s'.format(
				epoch, batch_idx * args.batch_size, len(private_train_loader) * args.batch_size,
				100. * batch_idx/len(private_train_loader), loss.item(), time.time()-start_time)) 

def test(args, model, private_test_loader):
	model.eval() 
	test_loss = 0 
	correct = 0 

	with torch.no_grad():
		for data, target in private_test_loader:
			start_time = time.time() 

			output = model(data) 
			pred = output.argmax(dim = 1) 
			correct += pred.eq(target.view_as(pred)).sum() 

	correct = correct.get().float_precision() 
	print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
		correct.item(), len(private_test_loader)*args.test_batch_size,
		100.*correct.itme()/(len(private_test_loader)*args.test_batch_size)))


# Launching the training 
model = Net() 
model = model.fix_precision().share(*workers, crypto_provider= crypto_provider, requires_grad = True) 

optimizer = optim.SGC(model, parameters(), lr = args.lr) 
optimizer = optimizer.fix_precision() 

for epoch in range(1, args.epochs +1):
	train(args, model, private_train_loader, optimizer, epoch) 
	tesst(args, model, private_test_loader) 

	








