"""
Day 14: Fri 12 Jul 2019 
1. Secure and Private AI Challenge
- I googled the term ‘differential privacy GitHub’ and could find out many people and companies are exploring the technique for the better deep learning of their own, and also could see many of them were enrolled students of our Challenge (https://github.com/topics/differential-privacy?o=asc&s=forks) 
- Out of the search results, deploying DP on CNN seemed pretty interesting, so I visited the repository and took a look (https://github.com/offthewallace/DP_CNN)
- The program was about implementing PATE(Private Aggregation of Teacher Ensembles) by using custom datasets and training CNN model on it.
- Wow, and I got to know about the repository Tensorflow holding, and the name of it is ‘Privacy’ and this contains research, tutorials, and the relevant codes for those. (https://github.com/tensorflow/privacy)
- And from the above, Tensorflow Privacy repository, took a look into ‘DP Logistic Regression on MNIST’ which was in the tutorials directory.
- Out of the different functions, tried to learn more about ‘lr_model_fn()’
- Hmmmm this is difficult to understand by far, but I am sure that I will get better and better! 
- Repo: https://github.com/marofmar/60daysofudacity/blob/master/Tensorflow_privacy_lr_DP_(Day14).py

"""

# MNIST
def lr_model_fn(features, labels, mode, nclasses, dim):
	input_layer = tf.reshape(features['x'], tuple([-1]) +dim)

	logits = tf.layers.dense(
		inputs = input_layer,
		units = nclasses, 
		kernel_regularizer = tf.contrib.layers.12_regularizer(
			scale = FLAGS.regularizer),
		biase_regularizer = tf.contrib.layers.12_regularizer(
			scale = FLAGS.regularizer))
	#calculate loss as a vector (to support microbatches in DP-SGD)
	vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels = labels, logits = logits) + tf.losses.get_regularization_loss()
	scalar_loss = tf.reduce_mean(vector_loss) 

	#configure the training op (for TRAIN mode) 
	if mode == tf.estimator.ModeKeys.TRAIN:
		if FLAGS.dpsgd:
			#hmmm
			# we don't use microbatches (thus speeding up computations) 
			# since no clipping is necessary due to the data normalization 
		optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
			l2_norm_clip = math.sqrt(2*(FLAGS.data_12_norm**2 +1)),
			nose_multiplier = FLAGS.noise_multiplier,
			num_microbaatches = 1,
			learning_rate = FLAGS.learning_rate)
		opt_loss = vector_loss 

	else:
		optimizer = GradientDescentOptimizer(learning_rate = FLAGS.learning_rate)
		opt_loss = scalar_loss
	global_step = tf.train.get_global_step()
	train_op = optimizer.minimize(loss=opt_loss, global_step = global_step) 

	# in the following, we paass the mean of the loss rather than the vecgor_loss
	# since tf.estimator requires a scalar loss (wow)
	# This is only used for evaluation and debugging by tf.estimator
	# The actual loss being minimized is opt_loss defined above and passed to optimier.minimize() 
	return tf.estimator.EstimatorSpec(
		mode = mode, loss = scalar_loss, train_op = train_op)

elif mode == tf.estimator.ModeKeys.EVAL:
	eval_metric_ops = {
	'accuracy':
	tf.metrics.accuracy(
		labels = labels, predictions = tf.argmax(input=logits, axis = 1))}
	return tf.estimator.EstimatorSpec(
		mode = mode, loss = scalar_loss, eval_metric_ops = eval_metric_ops)
	}
