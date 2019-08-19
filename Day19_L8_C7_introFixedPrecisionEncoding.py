# Goal: to aggregate gradients using the Secrit Sharing techniques 

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


