import tensorflow as tf
import pickle
from sqlite3 import dbapi2 as sqlite
import re
import numpy as np

# connetion to the database
con = sqlite.connect('barley_30m.db')
c = con.cursor()

images_train = np.array(
    [
        pickle.loads(img)
        for img, in c.execute('select rgb from img').fetchall()
    ]
)
labels_train_li = []
for label, in c.execute('select label from img').fetchall():
    if label == 0:
        labels_train_li.append([1, 0])
    else:
        labels_train_li.append([0, 1])
labels_train = np.array(labels_train_li)

'''
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
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)


# conv_1 layer
with tf.name_scope('conv-layer-1'):
    W_conv1 = weight_variable([5, 5, 3,32]) # outsize=32 :  convolutions units
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1) # 100 * 100 * 32
    h_pooled_1 = max_pool(h_conv1) # 25 * 25 * 32

# conv_2 layer
with tf.name_scope('conv-layer-2'):
    W_conv2 = weight_variable([5,5,32,64]) # outsize=64
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pooled_1, W_conv2) + b_conv2) # 25 * 25 *64
    h_pooled_2 = max_pool(h_conv2) # 25 * 25 * 64

# full-connection layer
with tf.name_scope('nn-layer-1'):
    W_fun1 = weight_variable([25*25*64, 1024])
    b_fun1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pooled_2, [-1, 25*25*64])
    h_fun2 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fun1) + b_fun1)
    h_fun2_drop = tf.nn.dropout(h_fun2, keep_prob)

# softmax layer
with tf.name_scope('nn-layer-2'):
    W_fun2 = weight_variable([1024, 2])
    b_fun2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fun2_drop, W_fun2) + b_fun2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction)))
train_step = tf.train.AdamOptimizer(1e-04).minimize(cross_entropy)

# accuracy
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

for i in range(20000):
    if i % 100 == 0:
        train_accuacy = accuracy.eval(feed_dict={xs: x_tr, ys: y_tr, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuacy))
    train_step.run(feed_dict = {xs: x_tr, y_: y_tr, keep_prob: 0.5})

# accuacy on test
print("test accuracy %g"%(accuracy.eval(feed_dict={xs: x_val, ys: y_val, keep_prob: 1.0})))
'''


def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.

    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        image = tf.random_crop(
            image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(
            image,
            target_height=img_size_cropped,
            target_width=img_size_cropped)

    return image


def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(
        lambda image: pre_process_image(image, training), images
    )

    return images


def random_batch(train_batch_size):
    # Number of images in the training-set.
    num_images = len(images_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch


def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch(train_batch_size=100)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # We also want to retrieve the global_step counter.
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # Print status to screen every 100 iterations (and last).
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

        # Save a checkpoint to disk every 1000 iterations (and last).
        if (i_global % 1000 == 0) or (i == num_iterations - 1):
            # Save all variables of the TensorFlow graph to a
            # checkpoint. Append the global_step counter
            # to the filename so we save the last several checkpoints.
            saver.save(session,
                       save_path=save_path,
                       global_step=global_step)

            print("Saved checkpoint.")

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif))))


def main_network(images, training):
    # Wrap the input images as a Pretty Tensor object.
    x_pretty = pt.wrap(images)

    # Pretty Tensor uses special numbers to distinguish between
    # the training and testing phases.
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    # Create the convolutional neural network using Pretty Tensor.
    # It is very similar to the previous tutorials, except
    # the use of so-called batch-normalization in the first layer.
    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=64, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=128, name='layer_fc2').\
            softmax_classifier(num_classes=num_classes, labels=y_true)

    return y_pred, loss


def create_network(training):
    # Wrap the neural network in the scope named 'network'.
    # Create new variables during training, and re-use during testing.
    with tf.variable_scope('network', reuse=not training):
        # Just rename the input placeholder variable for convenience.
        images = x

        # Create TensorFlow graph for pre-processing.
        images = pre_process(images=images, training=training)

        # Create TensorFlow graph for the main processing.
        y_pred, loss = main_network(images=images, training=training)

    return y_pred, loss


create_network(training=False)
