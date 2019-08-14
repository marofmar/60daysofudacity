'''
Due to real-life project deadlines, I had to skip a day or two. \
But, I will keep doing this 60 days of Udacity as my personal chanllenge goal!


Tue 13 Aug 2019 


1. Read Medium posting: https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b
2. Code implementation GAN START ! 
3. YouTube: Putting Humans at the Center of AI (https://youtu.be/Ev9BUGoOp64)

''' 


import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 

from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("./mnist/data/", one_hot = True) 

total_epoch = 100 
batch_size = 100 
learning_rate = 0.0001 

n_hidden = 256 
n_input = 28*28 
n_noise = 128 # noise size 

X = tf.placeholder(tf.float32, [None, n_input]) 
Z = tf.placeholder(tf.float32, [None, n_noise]) # noise 

# Generator 
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev = 0.01 ))
G_b1 = tf.Variable(tf.zeros([n_hidden])) 
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev = 0.01)) 
G_b2 = tf.Variable(tf.zeros([n_input])) 

# Discriminator 
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev = 0.01)) 
D_b1 = tf.Variable(tf.zeros([n_hidden])) 
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev = 0.01)) 
D_b2 = tf.Variable(tf.zeros([1])) # a singe scalar value telling how close sth is to the original 

# Generator NN 
def generator(noise_z):
	hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1) 
	output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2) 
	return output 

# Discriminator NN
def discriminator(inputs):
	hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1) 
	output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2) 
	return output 

# Make a rando noise Z 
def get_noise(batch_size, n_noise):
	return np.random.normal(size = (batch_size, n_noise)) 

# Using noise, make a random image  (random distribution in the proj perspective) 
G = generator(Z) 
# A value telling the generated image is real or not 
D_gene = discriminator(G) 
# A value based on the real one 
D_real = discriminator(X) 

# LOSS 
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1-D_gene)) 
loss_G = tf.reduce_mean(tf.log(D_gene)) 

# parmas for calculating loss 
D_var_list = [D_W1, D_b1, D_W2, D_b2] 
G_var_list = [G_W1, G_b1, G_W2, G_b2] 

# Optimization 
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list = D_var_list) 
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list = G_var_list) 

# Traninig NN 
sess = tf.Session() 
sess.run(tf.global_variables_initializer()) 

total_batch = int(mnist.train.num_examples/batch_size) 
loss_val_D, loss_val_G = 0, 0 

for epoch in range(total_epoch): 
	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size) 
		noise = get_noise(batch_size, n_noise) 

		_, loss_val_D = sess.run([train_D, loss_D], feed_dict = {X: batch_xs, Z: noise})
		_, loss_val_G = sess.run([train_G, loss_G], feed_dict = {Z: noise})

	if epoch == 0 or (epoch +1) %10 == 0:
		sample_size = 10
		noise = get_noise(sample_size, n_noise) 
		samples = sess.run(G, feed_dict = {Z: noise})

		fig, ax = plt.subplots(1, sample_size, figsize = (sample_size, 1)) 

		for i in range(sample_size):
			ax[i].set_axis_off() 
			ax[i].imshow(np.reshape(samples[i], (28,28)))

		#plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches = 'tight') 
		plt.close(fig) 

print('Finished.')













