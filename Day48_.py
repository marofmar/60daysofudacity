'''
found the error from the yesterday's VAE code!
-tf.reduce_mean(logpx_z + logpz - logqz_x)
I wrote wrongly as -tf.reduce_mean(logpx_z + logpz + logqz_x)

FIXED!

'''
optimizer = tf.keras.optimizers.Adam(1e-4) 

def log_normal_pdf(sample, mean, logvar, raxis = 1):
  log2pi = tf.math.log(2.*np.pi) 
  return tf.reduce_sum(
      -.5*((sample - mean) **2. * tf.exp(-logvar) + logvar + log2pi),
      axis = raxis)

@tf.function 
def compute_loss(model, x):
  mean, logvar = model.encode(x) 
  z = model.reparameterize(mean, logvar) 
  x_logit = model.decode(z) 
  
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits = x_logit, labels = x) 
  logpx_z = -tf.reduce_sum(cross_ent, axis = [1,2,3]) 
  logpz = log_normal_pdf(z, 0., 0.) 
  logqz_x = log_normal_pdf(z, mean, logvar) 
  return -tf.reduce_mean(logpx_z + logpz - logqz_x) # found the error!!!!!!!!!!!!!

@tf.function 
def compute_apply_gradients(model, x, optimizer):
  with tf.GradientTape() as tape: 
    loss = compute_loss(model, x) 
  gradients = tape.gradient(loss, model.trainable_variables) 
  optimizer.apply_gradients(zip(gradients, model.trainable_variables)) 