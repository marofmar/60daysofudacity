'''
Part 6: Federated Learning on MNIST using a CNN
- Federated Learning aims to build systems that learn on decentralized data, improving data privacy and ownership.

'''
# setting
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms 

import syft as sy #import PySyft library 
hook = sy.TorchHook(torch) # hook PyTorch 
bob = sy.VirtualWorker(hook, id = 'bob') 
alice = sy.VirtualWorker(hook, id = 'alice') 

class Arguments():
	def __init__(self):
		self.batch_size = 64 
		self.test_batch_size = 1000 
		self.epochs = 10 
		self.lr = 0.01 
		self.momentum = 0.5 
		self.no_cuda = False 
		self.seed = 1
		self.log_interval = 30 
		self.save_model = False 

args = Arguments() 

use_cuda = not args.no_cuda and torch.cuda.is_available() 

torch.manual_seed(args.seed) 

device = torch.device("cuda" if use_cuda else "cpu") 

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {} 

'''
daata loading and sending to workers 
- load the data and transform the training Dataset into a Federated Dataset split across the workers
- using the ".federate" method. 
- The federated dataset is now given to a Federated DataLoader and the test dataset remains unchaanged.
'''

federated_train_loader = sy.FederatedDataLoader(datasets.MNIST('../data', train = True, download = True,
	transform = transforms.Compose([transforms.ToTensor(), 
		transforms.Normalize((0.1307,),(0.3081,))])).federate((bob, alice)),batch_size=args.batch_size, shuffle = True, **kwargs)

test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train = False, transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,),(0.3081,))])),batch_size = args.test_batch_size, shuffle = True, **kwargs)

'''
CNN Specification
'''
class Net(nn.Module): 
	def __init__(self):
		super(Net, self).__init__() 
		self.conv1 = nn.Conv2d(1, 20, 5, 1) 
		self.conv2 = nn.Conv2d(20, 50, 5, 1) 
		self.fc1 = nn.Linear(4*4*50, 500) 
		self.fc2 = nn.Linear(500, 10) 

	def forward(self, x):
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2) 
		x = F.relu(self.conv2(x)) 
		x = F.max_pool2d(x, 2, 2) 
		x = x.view(-1, 4*4*50) 
		x = F.relu(self.fc1(x)) 
		x = self.fc2(x) 
		return F.log_softmax(x, dim = 1) 

'''
Define the train and test functions 
- Since the data batches are distributed across alice and bob, we need to send the model to the right location for each batch.
- After performing the operations remotely, we get back the model updated and the loss to look for improvement.

'''

def train(args, model, device, federated_train_loader, optimizer, epoch): 
	model.train() 
	for batch_idx, (data, target) in enumerate(federated_train_loader):
		model.send(data.location) # send the model to the right lcoation 
		data, target = daata.to(device), target.to(device) 
		optimizer.zero_grad() 
		output = model(data) 
		loss = F.nll_loss(output, target) 
		loss.backward() 
		optimizer.step() 
		model.get() # get the model back 

		if batch_idx % args.log_interval == 0:
			loss = loss.get() # get back the loss 
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:6f}'.format(epoch, baatch_idx * args.batch_size, len(federated_train_loader)*args.batch_size, 100*batch_idx/len(federated_trian_loader), loss.item()))

def test(args, model, device, test_loader):
	model.eval()
	total_loss = 0 
	correct = 0 
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device) 
			output = model(data) 
			test_loss += F.nll_loss(output, target, reduction = 'sum').item() 
			pred = output.argmax(1, keepdim = True) 
			correct += pred.eq(target.view_as(pred)).sum().item() 

	test_loss /= len(test_loader.dataset) 

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset), 100. *correct/len(test_loader.dataset)))

'''
Launching the training

'''
%%time
model = Net().to(device) 
optimizer = optim.SGD(model.parameters(), lr = aargs.lr) 

for epoch in range(1, args.epochs +1):
	train(args, model, device, federated_train_loader, optimizer, epoch) 
	test(args, model, device, test_loader) 

if (args.save_model):
	torch.save(model.state_dict(), "mnist_cnn.pt")





