#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:24:04 2020

@author: thorius
"""
import tensorflow as tf

class SWSA(tf.keras.layers.Layer):
    def __init__(self,
                 units = None,
                 weights_initializer='glorot_uniform',
                 weights_regularizer=None,
                 weights_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activation='relu',
                 **kwargs):
        super(SWSA, self).__init__(**kwargs)
        
        self.units = units
        self.weights_initializer = weights_initializer
        self.weights_constraint = weights_constraint
        self.weights_regularizer = weights_regularizer
        
        self.bias_initializer = bias_initializer
        self.bias_constraint = bias_constraint
        self.bias_regularizer =  bias_regularizer
        
        
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        super(SWSA, self).build(input_shape)
        
        feature_dim = input_shape[-1]
        
        self.attention_weights = self.add_weight(
            shape = (feature_dim, self.units),
            name = 'self_attention_weight',
            initializer=self.weights_initializer,
            constraint=self.weights_constraint,
            regularizer=self.weights_regularizer)
        
        
        
        self.bias = self.add_weight(
            shape = (self.units,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)
        
        
    def call(self, inputs):
        
        if inputs.shape.rank != 3:  
            raise ValueError('inputs.shape.rank: %d must be 3 ' % inputs.shape.rank)
        
        v = tf.matmul(inputs, self.attention_weights) + self.bias
        
        scaled_softmax_v = tf.nn.softmax(tf.matmul(v, tf.transpose(v, [0, 2, 1])) / self.units)
        attend_v = tf.matmul(scaled_softmax_v, v)
        
        attend_v = self.activation(attend_v)
        
        attend_v = tf.keras.layers.LayerNormalization(axis=2 , center=True , scale=True)(attend_v)
        return attend_v
        
    def get_config(self):
      config = {
          'weights_initializer': self.weights_initializer,
          'weights_regularizer': self.weights_regularizer,
          'weights_constraint': self.weights_constraint,
          'bias_initializer': self.bias_initializer,
          'bias_regularizer': self.bias_regularizer,
          'bias_constraint': self.bias_constraint,
      }
      base_config = super(SWSA, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))






































