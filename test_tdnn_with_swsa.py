#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 09:35:49 2020

@author: thorius
"""

from TDNN_layer import TDNNLayer
import numpy as np
import keras
import tensorflow as tf
from tdnn import TDNN
from swsa import SWSA
from keras.models import Sequential


batch = 1
input_context = [-2, 2]
context_size = input_context[-1] - input_context[0] + 1
time_size = 10
frequency_size = 3
output_size = 5

'''
input = tf.convert_to_tensor(np.random.randn(batch, time_size, frequency_size), 
                             dtype=tf.float32, 
                             name='input_tdnn')
print(input.shape)
print(input)

print("Sub-sampling:")
filter_sub = tf.ones([2, frequency_size , output_size])
output_sub = tf.nn.conv1d(input, filter_sub, stride=1, padding="VALID", dilations= context_size - 1)
print(output_sub.shape)
print(output_sub)
print("Sub-sampling:")
filter_unsub = tf.ones([context_size, frequency_size , output_size])
output_unsub = tf.nn.conv1d(input, filter_unsub, stride=1, padding="VALID", dilations= 1)
print(output_unsub.shape)
print(output_unsub)
'''

input_shape = (99, 40)
X_input = tf.keras.Input(input_shape)
X = TDNN(input_context = [-1, 1], units = 32, stride = 3, sub_sampling = False)(X_input)
X = SWSA(units = 32)(X)
X = TDNN(input_context = [-1, 1], units = 32, stride = 1, sub_sampling = False, pad = 'SAME')(X)
X = TDNN(input_context = [-1, 1], units = 32, stride = 1, sub_sampling = False, pad = 'SAME')(X)
X = tf.keras.layers.GlobalAveragePooling1D()(X)
X = tf.keras.layers.Dense(units = 11, use_bias=False)(X)
X = tf.nn.softmax(X)

model = tf.keras.models.Model(inputs=X_input, outputs=X, name='TDNN_v1')
model.summary()

'''
input_shape = (49, 30)
X_input = tf.keras.Input(input_shape)
X = TDNN(input_context = [-2, 2], units = 20, sub_sampling = False)(X_input)
X = TDNN(input_context = [-1, 2], units = 30, sub_sampling = False)(X)
model = tf.keras.models.Model(inputs=X_input, outputs=X, name='TDNN_v1')
model.summary()
'''