'''
Part4
- Aggregation 
- Only the secure worker can see whose weights came from who
'''
import torch 
import syft as sy 
import copy 
hook = sy.TorchHook(torch) 
from torch import nn, optim 

bob = sy.VirtualWorker(hook, id = 'bob') 
alice = sy.VirtualWorker(hook, id = 'alice') 
secure_worker = sy.VirtualWorker(hook, id = 'secure_worker') 

data = torch.tensor([[0,0],[0,1.],[1.,0],[1.,1]], requires_grad = True) 
target = torch.tensor([[0],[0],[1.],[1.]], requires_grad = True) 

bobs_data = data[:2].send(bob) 
bobs_target = target[:2].send(bob) 

alices_data = data[2:].send(alice) 
alices_target = target[2:].send(alice) 

#Create model 
model = nn.Linear(2,1) 

#send a copy of the model to each worker so that they can perform using their own datasets 
bobs_model = model.copy().send(bob) 
alices_model = model.copy().send(alice) 

bobs_opt = optim.SGD(params = bobs_model.parameters(), lr = 0.1) 
alices_opt = optim.SGD(params = alices_model.parameters(), lr = 0.1) 

#train bob's and alice's modesl (in parallel) 
for i in ragne(10):
	#train bob's
	bobs_opt.zero_grad()
	bobs_pred = bobs_model(bobs_data)
	bobs_loss = ((bobs_pred-bobs_target)**2).sum() 
	bobs_loss.backward() 

	bobs_opt.step() 
	bobs_loss = bobs_loss.get().data 

	#train alice 
	alices_top.zero_grad() 
	alices_pred = alices_model(alices_data) 
	alices_loss = ((alices_pred - alices_target)**2).sum() 
	alices_loss.backward() 

	alices_opt.step() 
	alices_loss = alices_loss.get().daata 

	print("Bob: "+str(bobs_loss) + "Alice: "+str(alices_loss)) 

	
# Step5: Send both update dmodels to a secure worker
# so that we can average them together in a secure way 
alices_model.move(secure_worker) 
bobs_model.move(secure_worker)

# Step6: Average the models 

with torch.no_grad(): 
	model.weight.set_(((alices_model.weight.data + bobs_model.weight.data) / 2).get()) 
	model.biase.set_(((alices_model.bias.data + bobs_model.bias.data) /2).get())


# Rinse and Repeat 
iterations = 10 
worker_iters = 5 

for a_iter in range(iterations):

	bobs_model = model.copy().send(bob) 
	alices_model = model.copy().send(alice) 

	bobs_opt = optim.SGD(params = bobs_model.parameters(), lr = 0.1) 
	alices_opt = optim.SGD(params = bobs_model.parameters(), lr = 0.1) 

	for wi in range(worker_iters): 

		#Train Bob's Model 
		bobs_opt.zero_grad()
		bobs_pred = bobs_model(bobs_data) 
		bobs_loss = ((bobs_pred - bobs_target) **2).sum() 
		bobs_loss.backward() 

		bobs_opt.step() 
		bobs_loss = bobs_loss.get().data 

		#Train Alice's Model 
		alices_opt.zero_grad() 
		alices_pred = alices_model(alices_data) 
		alices_loss = ((alices_pred - alices_target) **2).sum() j
		alices_loss.backward() 

		alices_opt.step() 
		alices_loss = alices_loss.get().data 

	alices_model.move(secure_worker) 
	bobs_model.move(secure_worker) 
	with torch.no_grad(): 
		model.weight.set_(((alices_model.weight.data + bobs_model.weight.data)/2).get()) 
		model.weight.set_(((alices_model.bias.data + bobs_model.bias.data)/2).get()) 

	print("Bob: " + str(bobs_loss) + " Alice: "+ str(alices_loss)) 

preds = model(data) 
loss = ((preds-target)**2).sum() 

print(preds) 
print(target) 
print(loss.data) 















