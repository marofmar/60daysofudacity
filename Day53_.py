'''
Day 53
- Studied C++ for the school class
- Studied Tensorflow Nueral Machine Translation Seq2Seq (https://github.com/tensorflow/nmt)

'''

def embedding():
	# Embedding
	embedding_encoder = variable_scope.get_variable(
	    "embedding_encoder", [src_vocab_size, embedding_size], ...)
	# Look up embedding:
	#   encoder_inputs: [max_time, batch_size]
	#   encoder_emb_inp: [max_time, batch_size, embedding_size]
	encoder_emb_inp = embedding_ops.embedding_lookup(
	    embedding_encoder, encoder_inputs)

	return

def encoder():
	# Build RNN cell
	encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

	# Run Dynamic RNN
	#   encoder_outputs: [max_time, batch_size, num_units]
	#   encoder_state: [batch_size, num_units]
	encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
	    encoder_cell, encoder_emb_inp,
	    sequence_length=source_sequence_length, time_major=True)
	return
def decoder():
	decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
	# Helper
	helper = tf.contrib.seq2seq.TrainingHelper(
	    decoder_emb_inp, decoder_lengths, time_major=True)
	# Decoder
	decoder = tf.contrib.seq2seq.BasicDecoder(
	    decoder_cell, helper, encoder_state,
	    output_layer=projection_layer)
	# Dynamic decoding
	outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
	logits = outputs.rnn_output
	return