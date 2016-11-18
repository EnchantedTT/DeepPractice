from __future__ import print_function
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_input = 784
n_h1 = 1024
n_h2 = 1024
n_o = 10

learning_rate = 0.01
train_steps = 100
batch_size = 100
display_step = 1

x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_o])

def mlp(x, weights, biases):
	h_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	h_1 = tf.nn.relu(h_1)

	h_2 = tf.add(tf.matmul(h_1, weights['h2']), biases['b2'])
	h_2 = tf.nn.relu(h_2)

	output = tf.matmul(h_2, weights['o']) + biases['o']

	return output

weights = {
	'h1':tf.Variable(tf.random_normal([n_input, n_h1])),
	'h2':tf.Variable(tf.random_normal([n_h1, n_h2])),
	'o':tf.Variable(tf.random_normal([n_h2, n_o]))
}

biases = {
	'b1':tf.Variable(tf.random_normal([n_h1])),
	'b2':tf.Variable(tf.random_normal([n_h2])),
	'o':tf.Variable(tf.random_normal([n_o]))
}

predict = mlp(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)

	#train
	for epoch in range(train_steps):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)

		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)

			_, c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})

			avg_cost += c / total_batch

		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
	print("Finish!")

	correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
