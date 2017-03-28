__author__ = 'socurites'

import tensorflow as tf
import numpy as np

# Load MNIST datastes
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# create variable for model: W * x + b
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# summary histogram for W
tf.summary.histogram('histogram', W)

# summary scalar for W
mean = tf.reduce_mean(W)
stddev = tf.sqrt(tf.reduce_mean(tf.square(W - mean)))
tf.summary.scalar('mean', mean)
tf.summary.scalar('stddev', stddev)
tf.summary.scalar('max', tf.reduce_max(W))
tf.summary.scalar('min', tf.reduce_min(W))

# define soft max funciton for output layer
y = tf.matmul(x, W) + b

# summary image for x
image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
tf.summary.image('input', image_shaped_input, 10)

# crate variable for real y
y_ = tf.placeholder(tf.float32, [None, 10])

# define loss: cross entropy
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# summary scalar for loss
tf.summary.scalar('cross_entropy', cross_entropy)

# define optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# merge all summary ops into a single op
merged = tf.summary.merge_all()

# run training repeatedly within session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# write summary to disk for TensorBoard visualization
train_writer = tf.summary.FileWriter('/home/itrocks/Downloads/train', sess.graph)

for i in range(2000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # run summary op and train op
  summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})

  # write summary events to disk
  train_writer.add_summary(summary, i)

train_writer.close()

# evaluate training accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# cast boolean into float
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))