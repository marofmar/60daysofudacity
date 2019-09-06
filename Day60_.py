'''
Day 60
LAST !

GAN to 1D data!!
'''


# disc add
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab 
%matplotlib inline
import tensorflow as tf 

sess = tf.InteractiveSession()

class GenerativeNetwork():
  dim_z = 1
  dim_g = 1 
  
  def __init__(self):
    rand_uni = tf.random_uniform_initializer(-1e1, 1e1) 
    self.z_input = tf.placeholder(tf.float32, shape = [None, self.dim_z], name = 'z-input') 
    self.w0 = tf.Variable(rand_uni([self.dim_z, self.dim_g])) 
    self.b0 = tf.Variable(rand_uni([self.dim_g])) 
    
    self.g = tf.nn.sigmoid(tf.matmul(self.z_input, self.w0) + self.b0) 
    
  def generate(self, z_i):
    g_i = sess.run([self.g], feed_dict = {self.z_input: z_i}) 
    return g_i[0] 
  
class Discriminator():
  dim_x = 1
  dim_d = 1 
  num_hidden_neurons = 10 
  
  def __init__(self):
    self.x_input = tf.placeholder(tf.float32, shape = [None, self.dim_x], name = 'x_input') 
    self.d_target = tf.placeholder(tf.float32, shape = [None, self.dim_d], name = 'd-target') 
    rand_uni = tf.random_uniform_initializer(-1e1, 1e1)
    self.w0 = tf.Variable(rand_uni([self.dim_x, self.num_hidden_neurons])) 
    self.b0 = tf.Variable(rand_uni([self.num_hidden_neurons])) 
    self.w1 = tf.Variable(rand_uni([self.num_hidden_neurons, self.dim_d])) 
    self.b1 = tf.Variable(rand_uni([self.dim_d])) 
    
    temp = tf.nn.sigmoid(tf.matmul(self.x_input, self.w0) + self.b0) 
    self.d = tf.nn.sigmoid(tf.matmul(temp, self.w1) + self.b1)
    
    self.loss = tf.losses.mean_squared_error(self.d, self.d_target) 
    
  def discriminate(self, x_i):
    d_i = sess.run([self.d], feed_dict = {self.x_input: x_i})
    return d_i[0]

def main():
  print('Hello, GAN!')
  mu = 0.8
  sigma = 0.1 
  num_samples = 100000 
  num_bins = 100
  
  x = np.random.normal(mu, sigma, num_samples)
  #print(x)
  z = np.random.uniform(0,1,num_samples) 
  g = np.ndarray(num_samples) 
  
  # network
  G = GenerativeNetwork() 
  D = Discriminator()
  
  # generate data
  tf.global_variables_initializer().run() 
  
#   for i in range(0, num_samples, 1):
#     z_i = np.reshape(z[i], (1, G.dim_z)) 
#     g[i] = G.generate(z_i)
  z_i = np.reshape(z, (num_samples, G.dim_z))
  g_i = G.generate(z_i) 
  g = np.reshape(g_i, (num_samples))
  
  #histogram
  bins = np.linspace(0,1,num_bins) 
  
  px, _ = np.histogram(x, bins = bins, density = True)
  pz, _ = np.histogram(z, bins = bins, density = True) 
  pg, _ = np.histogram(g, bins = bins, density = True)  
  v = np.linspace(0,1,len(px))
  
  v_i = np.reshape(v, (len(v), D.dim_x))
  db = D.discriminate(v_i) 
  db = np.reshape(db, len(v))
  
  l = plt.plot(v, px, 'b--', linewidth = 1)
  l = plt.plot(v, pz, 'r--', linewidth = 1)
  l = plt.plot(v, pg, 'g--', linewidth = 1) 
  l = plt.plot(v, db, 'k--', linewidth = 1) 
  
  plt.title('1D GAN Test') 
  plt.xlabel('Data Values') 
  plt.ylabel('Probability Density') 
  plt.show()
  plt.close()
  
if __name__ == '__main__':
  main()