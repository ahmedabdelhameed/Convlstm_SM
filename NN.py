# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:39:12 2020

@author: Hamada
"""
from keras.layers import Input, Dense
from keras.models import Model
from keras import utils
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


# Settings
encoding_dim = 32
num_classes = 10
# Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:2000]
y_train = y_train[:2000]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))



# The SAE # Multi task model
sae_input_img = Input(shape=(784,), name='input')
# Encoder: input to Z
encoded = Dense(256, activation='relu', name='encode_1')(sae_input_img)
encoded = Dense(128, activation='relu', name='encode_2')(encoded)
encoded = Dense(encoding_dim, activation='relu', name='z')(encoded)
# Classification: Z to class
predicted = Dense(num_classes, activation='softmax', name='class_output')(encoded)
# Reconstruction Decoder: Z to input
decoded = Dense(128, activation='relu', name='decode_1')(encoded)
decoded = Dense(256, activation='relu', name='decode_2')(decoded)
decoded = Dense(784, activation='sigmoid', name='reconst_output')(decoded)
# Take input and give classification and reconstruction
supervisedautoencoder = Model(inputs=[sae_input_img], outputs=[decoded, predicted])
supervisedautoencoder.compile(optimizer='SGD',
              loss={'class_output': 'sparse-categorical-crossentropy',
                    'reconst_output': 'binary_crossentropy'},
                     loss_weights={'class_output': 0.1, 'reconst_output': 1.0},
                     metrics=['acc'])
supervisedautoencoder.summary()




# Multi-Task Train
SAE_history = supervisedautoencoder.fit(x_train,
          {'reconst_output': x_train, 'class_output': y_train},
          epochs=350, batch_size=32, shuffle=True, verbose=1,
          validation_data=(x_test, {'reconst_output': x_test, 'class_output': y_test}))