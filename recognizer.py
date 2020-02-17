import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPool2D
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K

from sklearn.model_selection import train_test_split

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print('Tensorflow Version:', tf.__version__)
print('Keras Version:', keras.__version__)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('Train, Test set before reshape:', train.shape, test.shape)

y_train = train['label']
x_train = train.drop(labels=['label'], axis=1)
x_train = x_train/255.0
x_test = test/255.0
x_train = x_train.values.reshape(-1, 28, 28, 1)
x_test = x_test.values.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, num_classes=10)

print('Train, Test set after reshape:', train.shape, test.shape)

X = x_train
Y = y_train
x_train2, x_val2, y_train2, y_val2 = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

with tf.device('/device:GPU:0'):
    def cnn_model():
        model = Sequential([
            Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            BatchNormalization(),
            Convolution2D(filters=32, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            Convolution2D(filters=32, kernel_size=(5, 5), strides=2, padding='same', activation='relu'),
            BatchNormalization(),
            Dropout(0.4),

            Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            Convolution2D(filters=64, kernel_size=(5, 5), strides=2, padding='same', activation='relu'),
            BatchNormalization(),
            Dropout(0.4),

            Convolution2D(filters=128, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            Convolution2D(filters=128, kernel_size=(4, 4), strides=2, padding='same', activation='relu'),
            BatchNormalization(),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.4),
            Dense(10, activation='softmax')
        ])

        model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

cnn_model_final = cnn_model()
# cnn_model_final.summary()

learning_rate_reduction = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
checkpoint_cb = ModelCheckpoint('best_model_checkpoint.h5', save_best_only=True)
early_stopping = EarlyStopping(patience=40, restor_best_weights=True)

gen = ImageDataGenerator(rotation_range=10,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.3,
                         zoom_range=0.1)

epochs = 5
batch_size = 95
batches = gen.flow(x_train2, y_train2, batch_size = batch_size)
history = cnn_model_final.fit_generator(generator = batches,
                                        steps_per_epoch = x_train2.shape[0]//batch_size,
                                        validation_data = (x_val2, y_val2),
                                        epochs = epochs,
                                        callbacks = [checkpoint_cb, early_stopping, learning_rate_reduction])

cnn_history = history.history

fig, axs = plt.subplots(nrows=0, ncols=3, figsize=(30, 20))

loss = cnn_history['loss']
val_loss = cnn_history['val_loss']
accuracy = cnn_history['accuracy']
val_accuracy = cnn_history['val_accuracy']
lr = cnn_history['lr']
epoches = range(1, len(val_loss)+1)

axs[0]