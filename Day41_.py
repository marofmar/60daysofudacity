''''
Wed 14 Aug 2019
1. PyTorch DCGAN tutroial (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

2. GAN PyTorch yunjey github pytorch-tutorial posting
(https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/generative_adversarial_network/main.py)

'''

import os 
import torch 
import torchvision 
import torch.nn as nn 
from torchvision import transforms
from torchvision.utils import save_image 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# hyper parameters 
latent_size = 64 
hidden_size = 256 
image_size = 784 
num_epochs = 200 
batch_size = 100 
sample_dir = 'samples' 

# create directory if not exist 
if not os.path.exists(sample_dir):
	os.makedirs(sample_dir) 

# image processing 
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize()])