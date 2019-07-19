'''
Day 20: Fri 19 Jul 2019 @ Beijing
1. Secure and Private AI Challenge
- Review PATE analysis, found out I cannot recall the content by just reading and looking the codes…!  How important reviewing is in terms of studying new things!!! 
- repo: https://github.com/marofmar/60daysofudacity/blob/master/Day20_review_PATE.py
- OMG...? please,,, at leat github should work in here.
'''



import numpy as np 

num_teachers = 10 
num_examples = 10000
num_labels = 10 

preds = (np.random.rand(num_teachers, num_examples) * num_labels).astype(int).transpose(1,0) 
new_labels = list() 
for an_image in preds: 
	label_counts = np.bincount(an_image, minlength = num_labels) 
	epsilon = 0.1 
	beta = 1/ epsilon 

	for i in range(len(label_counts)):
		label_counts[i] += np.random.laplace(0, beta, 1) 

	nes_label = np.argmax(label_counts) 

	new_labels.append(new_label) 

labels = np.array([9,9,3,6,9,9,9,9,8,2]) 
counts = np.bincount(lables, minlength = 10) 
query_result = np.argmax(counts) 
print(query_result) 

'''
conda activate syft
pip install pysyft
pip install numpy
'''

from syft.frameworks.torch.differential_privacy import pate 


num_teachers, num_exmples, num_labels = (100,100,10) 
preds = (np.random.rand(num_teachers, num_examples)*num_labels).astype(int) 
indices = (np.random.rand(num_exmaples)*num_labels).astype(int)

data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds = preds, indices = indices, noise_eps = 0.1, delta = 1e-5) 
print("Data Independent Epsilon: ", data_ind_eps) 
print("Data Dependent Epsilon: ", data_dep_eps) 

preds[:, 0:5]*= 0 
data_dep_eps, data_ind_eps = pate.perform_anlysis(teacher_pred = preds, indices = indices, noise_eps = 0.1, delta = 1e-5) 
print("Data Independent Epsilon ", data_ind_eps) 
print("Data Dependent Epsilon: ", data_dep_eps) 

preds[:. 0:50] *= 0 
data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds = preds, indices = indices, noise_eps = 0.1, delta = 1e-5, moments = 20) 
prit("Data Independent Epsilon: ", data_ind_eps) 
print("Data Dependent Epsilon: ", data_dep_eps) 