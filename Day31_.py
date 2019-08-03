'''
Part7: Federated Learning with Federated Dataset 
- FederatedDataset class, which can be used like the PyTorch Dataset class
'''

import torch as th 
import syft as sy 
sy.create_sandbox(globals(), verbose = False) 

boston_data = grid.search("#boston", "#data", verbose = False, return_counter = False) 
boston_target = grid.search("boston", "target", verbose = False, return_counter = False) 

# load a model and an optimizer 
n_features = boston_data['alice'][0].shape[1]
n_targets = 1 
model = th.nn.Linear(n_features, n_targets) 
optimizer = th.optim.SGD(params = model.parameters(), lr = 0.0000001)

# cast the result in BaseDatasets
datasets = []
for worker in boston_data.keys():
	dataset = sy.BaseDataset(boston_data[worker][0], boston_target[worker][0])
	datasets.append(dataset)

# Build the FederatedDataset object 
dataset = sy.FederatedDataset(datasets) 
print(dataset.workers) 
# sandbox
# ['bob', 'theo', 'jason', 'alice', 'andy', 'jon']

# specify options for FederatedDataLoader
train_loader = sy.FederatedDataLoader(dataset, batch_size = 4, shuffle = False, drop_last = False) 

# iterate over epochs
epochs = 10 
for epoch in range(1, epochs +1):
	loss_accum = 0 
	for batch_idx, (data, target) in enumerate(train_loader):

		model.send(data.location) 

		optimier.zero_grad() 
		pred = model(data) 
		loss = ((pred-target)**2).sum() 
		loss.backward() 
		optimizer.step() 

		model.get() 
		loss = loss.get() 

		loss_accum += float(loss) 

		if batch_idx % 20 == 0:
			print('Train EpochP {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * data.shape[0], len(train_loader),
				100.*batch_idx/ len(train_loader), loss.item()))

	print('Total Loss', loss_accum)








