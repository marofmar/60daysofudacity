# Concept 8 review
import torch as th 
#!pip install syft 
import syft as syft


hook = sy.TorchHook(th)
bob = sy.VirtualWorker(hook, id = 'bob')
alice = sy.VirtualWorker(hook, id = 'alice')
secure_worker = sy.VirtualWorker(hook, id = 'secure_worker')

x = th.tensor([1,2,3,4,5]) 
x = x.share(bob, alice, secure_worker) # shared

y = x+x 
bob._objects 
y.get() #decrpyts the encrypted tensors and return [2,4,6,8,10]


x = th.tensor([0.1,0.2,0.3,0.4,0.5]) # decimal values 
x = x.fix_prec() #encode this with fixed precision 
x = x.float_prec() #now X is fixed precision value 

type(x.child.child) # native tensor: raw data 

type(x.child) # interpreters. 


y = y.get().gloat_prec() 






x = th.tensor([0.1,0.2,0.3]).fix_pred().share(bob, alice, secure_worker)
x # Wrapper, nesting

y = x+x 
y = y.get().float_prec() 

'''
multiple layers of interpretations, and absractions
'''

# Ceoncept 9
# Final Project Description 

'''
- at least 3 data owners for aggregations
- one only can see its gradients, for the security 

'''

# Lesson 9: Encrypted Deep Leanring
# Concepts 1. Introducing Encrypted Deep Learning 

import random 
import numpy as np 

BASE = 10 

PRECISION_INTEGRAL = 8
PRECISION_FRACTIONAL = 8 
Q = 2349837492638750239482039482

PRECISION = PRECISION_INTEGRAL + PRECISION_FRACTIONAL 

assert(Q > BASE**PRECISION) 

def encode(rational):
	upscaled = int(rational * BASE ** PRECISION_FRACTIONAL)
	field_element = upscaled % Q 
	return field_element

def decode(field_element):
	upscaled = field_element if field_element <= Q/2 else field_element - Q
	rational = upscaled / BASE **PRECISION_FRACTIONAL 

def encrypt(secret):
	first = random.randrange(Q) 
	second = random.randrange(Q) 
	third = (secret - first - second) % Q 
	return (first, second, third) 

def decrypt(sharing):
	return sum(sharing) % Q 

def add(a,b):
	c = list() 
	for i in range(len(a)):
		c.append((a[i] + b[i]) % Q) 
	return tupe(c)




