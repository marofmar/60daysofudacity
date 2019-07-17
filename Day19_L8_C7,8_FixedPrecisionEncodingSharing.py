"""
Day 19: Thu 18 Jul 2019 
1. Secure and Private AI Challenge
- Watched Webinar (around 60%) 
	- Just dive in.
	- You are not your code. 
	- By just using the code, also a contribution, (since you could find some bugs that the developers could think of before!) 
	- How your code work in others’ devices and how those would communicate with other devices 
- Lesson8 Concept7: Intro to Fixed Precision Encoding 
- Lesson8 Concept8: Secret Sharing and Fixed Precision in PySyft 
"""


# Goal: to aggregate gradients using the Secrit Sharing techniques 

# Concept 7
BASE = 10 
PRECISION = 4 
Q = 234837492834203846


def encode(x_dec):
	return int(x_dec * (BASE** PRECISION)) % Q 

print(encode(0.5)) 

def decode(x_fp):
	return (x_fp if x_fp <= Q/2 else x_fp - Q) /BASE**PRECISION

print(decode(encode(0.5))) 

decode(5000+5000) 


# Concept 8
import torch as th 
#!pip install syft 
import syft as syft


hook = sy.TorchHook(th)
bob = sy.VirtualWorker(hook, id = 'bob')
alice = sy.VirtualWorker(hook, id = 'alice')
secure_worker = sy.VirtualWorker(hook, id = 'secure_worker')

x = th.tensor([1,2,3,4,5]) 
x = x.share(bob, alice, secure_worker) 

y = x+x 
bob._objects 
y.get() 


x = th.tensor([0.1,0.2,0.3,0.4,0.5])
x = x.fix_prec() 
x = x.float_prec() 

type(x.child.child) # ohhhh child child...?!

type(x.child) # child?!!!


y = y.get().gloat_prec() 




