import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="W")


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="bias")


def conv2d(x, W):
    return tf.nn.conv2d(
        x, W, strides=[1,1,1,1], padding="SAME", name="conv2d")


def max_pool(x):
    return tf.nn.max_pool(
        x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name="pooled")

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28,28, 1])

# conv_1 layer
with tf.name_scope('conv-layer-1'):
    W_conv1 = weight_variable([5,5,1,32]) # outsize=32 :  convolutions units
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # 28 * 28 * 32
    h_pooled_1 = max_pool(h_conv1) # 14*14*32

# conv_2 layer
with tf.name_scope('conv-layer-2'):
    W_conv2 = weight_variable([5,5,32,64]) # outsize=64
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pooled_1, W_conv2) + b_conv2) # 14 * 14 *64
    h_pooled_2 = max_pool(h_conv2) # 7 * 7 * 64

# func1 layer
with tf.name_scope('nn-layer-1'):
    W_fun1 = weight_variable([7*7*64, 1024])
    b_fun1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pooled_2, [-1, 7*7*64])
    h_fun2 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fun1) + b_fun1)
    h_fun2_drop = tf.nn.dropout(h_fun2, keep_prob)

# func2 layer
with tf.name_scope('nn-layer-2'):
    W_fun2 = weight_variable([1024, 10])
    b_fun2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fun2_drop, W_fun2) + b_fun2)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction)))
train_step = tf.train.AdamOptimizer(1e-04).minimize(cross_entropy)

## accuracy
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
import time
n_epochs = 15
batch_size = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    st = time.time()
    for epoch in range(n_epochs):
        n_batch = mnist.train.num_examples / batch_size
        for i in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob:0.6})
        print 'epoch', 1+epoch, 'accuracy:', sess.run(accuracy, feed_dict={keep_prob:1.0, xs: mnist.test.images, ys: mnist.test.labels})
    end = time.time()
    print '*' * 30
    print 'training finish.\ncost time:', int(end-st) , 'seconds;\naccuracy:', sess.run(accuracy, feed_dict={keep_prob:1.0, xs: mnist.test.images, ys: mnist.test.labels})
