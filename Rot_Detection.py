import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from keras.models import load_model

train_path = '/Users/krish/Dataset/dataset/train'
test_path = '/Users/krish/Dataset/dataset/test'
BATCH_SIZE = 10


def RotDetection(model_name):
    d = {
        'vgg16': tf.keras.applications.vgg16.preprocess_input,
        'resnet': tf.keras.applications.resnet.preprocess_input,
        'nasnet': tf.keras.applications.nasnet.preprocess_input,
        'xception': tf.keras.applications.xception.preprocess_input,
        'resnet50': tf.keras.applications.resnet50.preprocess_input,
        'resnetv2': tf.keras.applications.resnet_v2.preprocess_input,
        'densenet': tf.keras.applications.densenet.preprocess_input,
        'vgg19': tf.keras.applications.vgg19.preprocess_input

    }
    preprofunc = d[str(model_name)]
    train_batches = ImageDataGenerator(
        preprocessing_function=preprofunc,
        rescale=1 / 255.,
        horizontal_flip=True,
        vertical_flip=True
    ).flow_from_directory(
        directory=train_path,
        target_size=(20, 20),
        classes=['freshapples', 'freshbananas', 'freshoranges', 'rottenapples', 'rottenbananas', 'rottenorganges'],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb'
    )
    test_batches = ImageDataGenerator(
        preprocessing_function=preprofunc, rescale=1 / 255.
    ).flow_from_directory(
        directory=test_path,
        target_size=(20, 20),
        classes=['freshapples', 'freshbananas', 'freshoranges', 'rottenapples', 'rottenbananas', 'rottenorganges'],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation=('relu'), input_shape=(20, 20, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation=('relu')))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation=('relu')))
    model.add(Dense(128, activation=('relu')))
    model.add(Dense(6, activation=('softmax')))
    # evaluating the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_batches, epochs=17)