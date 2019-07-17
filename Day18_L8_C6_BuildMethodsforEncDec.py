import random 
Q = 234957493278467837892
x = 5

# def encrypt(x, n_shares = 3):
n_shares = 3
shares = list() 

for i in range(n_shares -1):
	shares.append(random.randint(0,Q)) 

final_share = Q-(sum(shares)%Q)+x 
shares.append(final_share)
sum(shares) 

def encrypt(x, n_shares = 3):
	n_shares = 3
	shares = list() 

	for i in range(n_shares -1):
		shares.append(random.randint(0,Q)) 

	final_share = Q-(sum(shares)%Q) + x 

	shares.append(final_share) 

	return tuple(shares) 

# run!
encrypt(5,)
encrypt(5, n_shares = 10) 

# now decrypt 

def decrypt(shares):
	return sum(shares) % Q 


def add(a, b):

	c = list() 
	
	assert(len(a) == len(b))

	for i in range(len(a)):
		c.append((a[i]+b[i])%Q) 

	return tuple(c)

decrypt(add(encrypt(5), encrypt(10))) # 15

x = encrypt(5)
y = encrypt(10)
z = add(x, y)
print(z) #sth not read-able 3 numbers 
decrypt(z) # 15 



