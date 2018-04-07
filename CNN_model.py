from __future__ import absolute_import, division, print_function

import numpy as np
import pickle
import cv2
import os
import shutil
from sqlite3 import dbapi2 as sqlite

import keras
from keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array
)
from keras.models import Sequential
from keras.layers import (
    Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D,
    GlobalAveragePooling2D, AveragePooling2D
)
from keras import optimizers
from keras.initializers import RandomNormal
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model

batch_size    = 200
epochs        = 10
iterations    = 100
dropout       = 0.25
weight_decay  = 0.0001
log_filepath  = './logs'


def scheduler(epoch):
    if epoch <= 30:
        return 0.005
    if epoch <= 60:
        return 0.005
    return 0.001


def check_logs():
    if os.path.exists('./logs'):
        if len(os.listdir('./logs')) == 0:
            return
        else:
            shutil.rmtree('./logs/')
            return


def build_model():
    # build the network
    model = Sequential()

    model.add(
        Conv2D(
            64,
            (3, 3),
            padding='same',
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            kernel_initializer="he_normal",
            input_shape=x_train.shape[1:]
        )
    )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

    model.add(
        Conv2D(
            128,
            (3, 3),
            padding='same',
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            kernel_initializer="he_normal"))
    model.add(Activation('relu'))
    model.add(
        Conv2D(
            128,
            (3, 3),
            padding='same',
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            kernel_initializer="he_normal"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding = 'same'))

   #model.add(
   #    Conv2D(
   #        256,
   #        (3, 3),
   #        padding='same',
   #        kernel_regularizer=keras.regularizers.l2(weight_decay),
   #        kernel_initializer="he_normal"))
   #model.add(Activation('relu'))
   #model.add(
   #    Conv2D(
   #        256,
   #        (3, 3),
   #        padding='same',
   #        kernel_regularizer=keras.regularizers.l2(weight_decay),
   #        kernel_initializer="he_normal"))
   #model.add(Activation('relu'))
   #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding = 'same'))

    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    # initiate RMSprop optimizer
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # Let's train the model using RMSprop
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy'])

    return model

if __name__ == '__main__':
    # connection to the database
    con = sqlite.connect('barley_30m.db')
    c = con.cursor()

    training_size = 3000
    ids = [i for i, in c.execute('select id from img').fetchall()]
    id_train = np.random.choice(ids, size=training_size)
    id_validation = [i for i in ids if i not in id_train]

    img_train = []
    img_val = []
    label_train = []
    label_val = []

    for i in id_train:
        rgb, label = c.execute('select rgb, label from img where id=%s' % i).fetchone()
        img_train.append(pickle.loads(rgb))
        if label == 1:
            label_train.append(1)
        else:
            label_train.append(0)

    x_train = np.array(img_train)
    y_train = np.array(label_train)

    for i in id_validation:
        rgb, label = c.execute('select rgb,label from img where id=%s'
                               %i).fetchone()
        img_val.append(pickle.loads(rgb))
        if label == 1:
            label_val.append(1)
        else:
            label_val.append(0)

    x_val = np.array(img_val)
    y_val = np.array(label_val)


    num_classes = len(set(y_train))

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    print('Totally {} classes'.format(num_classes))

    print('{} samples will be trained'.format(len(y_train)))

    # build network
    model = build_model()
    print(model.summary())
    # if you want to plot the model framework
    # plot_model(model=model, to_file='./model.png', show_shapes=True)

    # set callback
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr,tb_cb]

    # set data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True, # randomly flip images
    )

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # check_logs
    check_logs()

    # start training
    model.fit_generator(
        datagen.flow(
            x_train,
            y_train,
            batch_size=batch_size
        ),
        steps_per_epoch=iterations,
        epochs=epochs,
        callbacks=cbks,
        validation_data=(x_val, y_val),
    )
    model.save('model.h5')
