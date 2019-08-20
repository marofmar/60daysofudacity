'''
Day 45
Lesson 9, lecture 3,4,5
'''

import string 

char2index = {}
index2char = {} 

for i , char in enumerate(' '+string.ascii_lowercase + '0123456789' + string.punctuation ):
	char2index[char] = i 
	index2char[i] = char 

def string2values(str_input, max_len = 8):
	str_input = str_input[:max_len].lower() 
	if (len(str_input) < max_len):
		str_input = str_input + "."*(max_len-len(str_input)) 

	values = list() 
	for char in str_input: 
		values.append(char2index[char])
	return th.tensor(values).long() 

def one_hot(index, length):
	vect = th.zeros(length).long() 
	vect[index] = 1 
	return vect 

def string2one_hot_matrix(str_input):
	str_input = str_input[:max_len].lower() 
	if (len(str_input) < max_len):
		str_input = str_input + "."*(max_len-len(str_input)) 
	char_vectors = list() 
	for char in str_input:
		char_v = one_hot(char2index[char], len(index2char)).unsqueeze(0) 
		char_vectors.append(char_v) 
	return th.cat(char_vectors, dim = 0) 




