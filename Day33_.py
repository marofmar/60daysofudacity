'''
Part9: Intro to Encrypted Programs
'''

#Step1: Encryption using Secure Multi-Party Computation 

'''
Encrypt()
Encryption does not use floats or real numbers 
but happens in a mathematical space called 'integer quotient ring'
which is basically the integer btw 0 and Q-1, where Q is prime and 'big enough' 
so that the space can contain all the numbers that we use in our experiments.
In practice, given a value x integer, we do x%Q to fit in the ring.
'''
Q = 1234567891011
x = 25 
import random
def encrypt(x):
	share_a = random.randint(0,Q)
    share_b = random.randint(0,Q)
    share_c = (x-share_a-share_b)%Q 
    return (share_a, share_b, share_c)

encrypt(x) 

# Decrypt()
def decrypt(*shares):
	return sum(shares)%Q 

a,b,c = encrypt(25) 
decrypt(a,b,c)
decrypt(a,b) # wrong. so, need whole bunch of shares to decrypt properly.

# Step 2: Basic Arithmetic Using SMPC 
x = encrypt(25)
y = encrypt(5) 

def add(x, y):
	z = list() 
	# the first worker adds their shares together
	z.append((x[0] + y[0])%Q)

	# the second worker adds their shares together
	z.append((x[1] + y[1])%Q) 

	# the third worker dds their shares together 
	z.append((x[2] + y[2])%Q) 

decrypt(*add(x,y)) # works well 

# Step3: SMPC Using PySyft 
import torch 
import syft as sy 
hook = sy.TorchHook(torch) 

bob = sy.VirtualWorker(hook, id = 'bob') 
alice = sy.VirtualWorker(hook, id = 'alice')
bill = sy.VirtualWorker(hook, id = 'bill')

# Basic Encryption Decryption 
x = torch.tensor([25]) 
encrypted_x = x.share(bob, alice, bill) 
encrypted_x.get() # tensor[25]

#Introspecting the Encrypted Values 
bob._objects # {} empty 

x = torch.tensor([25]).share(bob, alice, bill) 

bobs_share = list(bob._objects.values())[0] 
alices_share = list(alice._objects.values())[0] 
bills_share = list(bill._objects.values())[0] 
print(bobs_share, alices_share, bills_share) 

Q = x.child.field 

(bobs_share + alices_share + bills_share) % Q 

# Encrypted Arithmetic 
x = torch.tensor([25]).share(bob, alice) 
y = torch.tensor([5]).share(bob, alice) 

z = x + y 
z.get() # tensor([30]) 
z = x - y 
z.get() # tensor([20]) 

# Encrypted Multiplication 
crypto_provider = sy.VirtualWorker(hook, id = 'crypto_provider') # additional someone who needs to be trusted to not collude with exsiting shareholders

x = torch.tensor([25]).share(bob, alice, crypto_provider = crypto_provider) 
y = torch.tensor([5]).share(bob, alice, crypto_provider = crypto_provider) 

z = x * y 
z.get() 
#matrix multiplication
x = torch.tensor([[1,2],[3,4]]).share(bob, alice, crypto_provider = crypto_provider)
y = torch.tensor([[2,0],[0,2]]).share(bob, alice, crypto_provider = crypto_provider) 

z = x.mm(y) 
z.get() 

#Encrypted Comparison , rely on SecureNN Protocol 
x = torch.tensor([25]).share(bob, alice, crypto_provider = crypto_provider) 
y = torch.tensor([5]).share(bob, alice, crypto_provider = crypto_provider)

z = x > y 
z.get() #tensor([1])

z = x <= y 
z.get() #tensor([0])

z = x == y 
z.get() #tensor([0])

z = x == y + 20 
z.geT() #tensor([1]) 

x = torch.tensor([2,3,4,1]).share(bob, alice, crypto_provider = crypto_provider)
x.max().get() #tensor([4]) 

x = torch.tensor([[2,3],[4,1]]).share(bob, alice, crypto_provider = crypto_provider)
max_values, max_idx = x.max(dim = 0) 
max_values.get() 




