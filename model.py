#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 21:06:25 2020

@author: thorius
"""

import tensorflow as tf
from tdnn import TDNN
from swsa import SWSA
from mfcc import MFCC
from attend import Attend


tf.keras.backend.clear_session()

num_samples = 16000
input_shape = (num_samples)
X_input = tf.keras.Input(input_shape)
X = MFCC()(X_input)

X = TDNN(input_context = [-1, 1], units = 32, stride = 3, sub_sampling = False)(X)
X = tf.keras.layers.BatchNormalization(axis = -1, trainable = False)(X)


#X = SWSA(units = 32, mode = 'multi-head', num_head = 4)(X)
X = SWSA(units = 32, mode = 'scaled-dot')(X)

X = TDNN(input_context = [-1, 1], units = 32, stride = 1, sub_sampling = False, pad = 'SAME')(X)
X = tf.keras.layers.BatchNormalization(axis = -1, trainable = False)(X)
X = TDNN(input_context = [-1, 1], units = 32, stride = 1, sub_sampling = False, pad = 'SAME')(X)
X = tf.keras.layers.BatchNormalization(axis = -1, trainable = False)(X)
X = tf.keras.layers.GlobalAveragePooling1D()(X)
X = tf.keras.layers.Dense(units = 11, use_bias= False)(X)
X = tf.nn.softmax(X)

model = tf.keras.models.Model(inputs=X_input, outputs=X, name='TDNN_with_shared_weight_self_attention_v1')
model.summary()