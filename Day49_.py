def random_crop(image):
	cropped_image = tf.image.random_crop(image, size = [IMG_HEIGHT, IMG_WIDTH, 3]) 
	return cropped_image

def normalize(image):
	image = tf.cast(image, tf.float32) 
	image = (image/127.5)- 1
	return image

def random_jitter(image):
	image = tf.image.resize(image, [286, 286], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
	image = random_crop(image) 
	image = tf.image.random_flip_left_right(image) 
	return image 

def preprocess_image_train(image, label):
	image = random_jitter(image) 
	image = normalize(image) 
	return image 

def preprocess_image_test(image, label):
	image= normalize(image) 
	return image 

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits = True) 

def discriminator_loss(real, generated):
	real_loss = loss_obj(tf.ones_like(real), real)
	generated_loss = loss_obj(tf.zeros_like(generated), generated) 
	tatal_disc_loss = real_loss + generated_loss 
	return total_disc_loss * 0.5 

def generator_loss(generated):

LAMBDA = 10 

def calc_cycle_loss(real_image, cycled_image):
	loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image)) 
	return LAMBDA * loss1 

def identity_loss(real_image, same_image):
	loss = tf.reduce_mean(tf.abs(real_image - same_image)) 
	return LAMBDA * 0.5 * loss 
