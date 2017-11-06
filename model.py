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
        label_train.append([1])
    else:
        label_train.append([0])

x_tr = np.array(img_train)
y_tr = np.array(label_train)

for i in id_validation:
    rgb, label = c.execute('select rgb, label from img where id=%s' % i).fetchone()
    img_val.append(pickle.loads(rgb))
    if label == 1:
        label_val.append([1])
    else:
        label_val.append([0])

x_val = np.array(img_val)
y_val = np.array(label_val)

sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="W")


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="bias")


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME", name="conv2d")


def max_pool(x):
    return tf.nn.max_pool(
        x,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        name="pooled")

xs = tf.placeholder(tf.float32, [None, 100, 100, 3])
ys = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32)


# conv_1 layer
# with tf.name_scope('conv-layer-1'):
W_conv1 = weight_variable([5, 5, 3, 16]) # outsize=32 :  convolutions units
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1) # 100 * 100 * 32
h_pooled_1 = max_pool(h_conv1) # 50 * 50 * 32


# conv_2 layer
#with tf.name_scope('conv-layer-2'):
W_conv2 = weight_variable([5,5,16,8]) # outsize=64
b_conv2 = bias_variable([8])
h_conv2 = tf.nn.relu(conv2d(h_pooled_1, W_conv2) + b_conv2) # 25 * 25 *64
h_pooled_2 = max_pool(h_conv2) # 25 * 25 * 64


# func1 layer
# with tf.name_scope('nn-layer-1'):
W_fun1 = weight_variable([25*25*8, 1024])
b_fun1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pooled_2, [-1, 25*25*8])
h_fun2 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fun1) + b_fun1)
h_fun2_drop = tf.nn.dropout(h_fun2, keep_prob)


# func2 layer
# with tf.name_scope('nn-layer-2'):
W_fun2 = weight_variable([1024, 1])
b_fun2 = bias_variable([1])
# softmax
# prediction = tf.nn.softmax(tf.matmul(h_fun2_drop, W_fun2) + b_fun2)
# logic regression
prediction = tf.nn.sigmoid(tf.matmul(h_fun2_drop, W_fun2) + b_fun2)
loss =  tf.square(prediction - ys, name="loss")
print(prediction)

cost_function = tf.reduce_mean(tf.reduce_sum((-ys * tf.log(prediction)) - ((1 - ys) * tf.log(1 - prediction))))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction)))
# train_step = tf.train.AdamOptimizer(1e-04).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-04).minimize(cost_function)
# accuracy
# correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
correct_prediction = tf.equal(tf.round(prediction), ys)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()

for i in range(10):
    if i % 10 == 0:
        train_accuacy = accuracy.eval(feed_dict={xs: x_tr, ys: y_tr, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuacy))
        print("test accuracy %g"%(accuracy.eval(feed_dict={xs: x_val, ys: y_val, keep_prob: 1.0})))
    train_step.run(feed_dict = {xs: x_tr, ys: y_tr, keep_prob: 1.0})

# accuacy on test
print("test accuracy %g"%(accuracy.eval(feed_dict={xs: x_val, ys: y_val, keep_prob: 1.0})))


save_path = saver.save(sess, './model.ckpl')
