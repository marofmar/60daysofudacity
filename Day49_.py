def random_crop(image):
	cropped_image = tf.image.random_crop(image, size = [IMG_HEIGHT, IMG_WIDTH, 3]) 
	return cropped_image

def normalize(image):
	image = tf.cast(image, tf.float32) 
	image = (image/127.5)- 1
	return image