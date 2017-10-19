import tensorflow as tf
import pickle
from sqlite3 import dbapi2 as sqlite
import re
import numpy as np
from tensorflow.python.ops import control_flow_ops

# connetion to the database
con = sqlite.connect('barley_30m.db')
c = con.cursor()

ids = [i for i, in c.execute('select id from img').fetchall()]
id_train = np.random.choice(ids, size=50)
id_validation = [i for i in ids if i not in id_train]

img_train = []
img_val = []
label_train = []
label_val = []

for i in id_train:
    rgb, label = c.execute('select rgb, label from img where id=%s' % i).fetchone()
    img_train.append(pickle.loads(rgb))
    if label == 1:
        label_train.append([0, 1])
    else:
        label_train.append([1, 0])

x_tr = np.array(img_train)
y_tr = np.array(label_train)

for i in id_validation:
    rgb, label = c.execute('select rgb, label from img where id=%s' % i).fetchone()
    img_val.append(pickle.loads(rgb))
    if label == 1:
        label_val.append([0, 1])
    else:
        label_val.append([1, 0])

x_val = np.array(img_val)
y_val = np.array(label_val)

sess = tf.InteractiveSession()

def conv_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)
    beta = tf.get_variable('beta', [n_out], initializer=beta_init)
    gamma = tf.get_variable('gamma', [n_out], initializer=gamma_init)

    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')

    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean = ema.average(batch_mean)
    ema_var = ema.average(batch_var)

    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = control_flow_ops.cond(
        phase_train,
        mean_var_with_update,
        lambda: (ema_mean, ema_var))

    normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, True)

    return normed

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="W")


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="bias")


def conv2d(x,weight_shape, bias_shape, phase_train, visualize=False):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
    W = tf.get_variable('W', weight_shape, initializer=weight_init)
    if visualize:
        filter_summary(W, weight_shape)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable('b', bias_shape, initializer=bias_init)
    logits = tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME'), b)

    return tf.nn.relu(conv_batch_norm(logits, weight_shape[3], phase_train))


def max_pool(x):
    return tf.nn.max_pool(
        x,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        name="pooled")

xs = tf.placeholder(tf.float32, [None, 100, 100, 3])
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
phase_train = tf.placeholder(tf.bool)


# conv_1 layer
# with tf.name_scope('conv-layer-1'):
h_conv1 = conv2d(xs, [5, 5, 3, 64], [64], phase_train) # 100 * 100 * 32
h_pooled_1 = max_pool(h_conv1) # 50 * 50 * 32


# conv_2 layer
#with tf.name_scope('conv-layer-2'):
h_conv2 = conv2d(h_conv1, [5, 5, 3, 64], [64], phase_train) # 50 * 50 *64
h_pooled_2 = max_pool(h_conv2) # 25 * 25 * 64


# conv_3 layer
#with tf.name_scope('conv-layer-2'):
h_conv3 = conv2d(h_conv2, [5, 5, 3, 64], [64], phase_train) # 25 * 25 *64
h_pooled_3 = max_pool(h_conv3) # 13 * 13 * 128


# func1 layer
# with tf.name_scope('nn-layer-1'):
W_fun1 = weight_variable([13*13*64, 2048])
b_fun1 = bias_variable([2048])
h_pool2_flat = tf.reshape(h_pooled_3, [-1, 13*13*64])
h_fun2 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fun1) + b_fun1)
h_fun2_drop = tf.nn.dropout(h_fun2, keep_prob)


# func2 layer
# with tf.name_scope('nn-layer-2'):
W_fun2 = weight_variable([2048, 2])
b_fun2 = bias_variable([2])
prediction = tf.nn.softmax(tf.matmul(h_fun2_drop, W_fun2) + b_fun2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction)))
train_step = tf.train.AdamOptimizer(1e-04).minimize(cross_entropy)

# accuracy
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

for i in range(2000):
    if i % 10 == 0:
        train_accuacy = accuracy.eval(feed_dict={xs: x_tr, ys: y_tr, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuacy))
    train_step.run(feed_dict = {xs: x_tr, ys: y_tr, keep_prob: 1.0})

# accuacy on test
print("test accuracy %g"%(accuracy.eval(feed_dict={xs: x_val, ys: y_val, keep_prob: 1.0})))
