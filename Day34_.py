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
	