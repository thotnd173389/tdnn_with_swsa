#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 10:30:36 2020

@author: thorius
"""

import tensorflow as tf


class TDNN(tf.keras.layers.Layer):
    def __init__(self,
                 input_context = [-2, 2],
                 sub_sampling = False,
                 units= None,
                 stride = 1,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=False,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activation='relu',
                 pad='VALID',
                 **kwargs):
        
        super(TDNN, self).__init__(**kwargs)
        self.input_context = input_context
        self.sub_sampling = sub_sampling
        self.units = units
        
        self.stride = stride
        
        self.kernel_initializer = kernel_initializer
        self.kernel_constraint = kernel_constraint
        self.kernel_regularizer = kernel_regularizer
        
        self.use_bias = use_bias
        
        self.bias_initializer = bias_initializer
        self.bias_constraint = bias_constraint
        self.bias_regularizer = bias_regularizer
        
        self.activation = tf.keras.activations.get(activation)
        
        self.pad = pad
        
        
    def build(self, input_shape):
        super(TDNN, self).build(input_shape)
        
        feature_dim = input_shape[-1]
        context_size = self.input_context[-1] - self.input_context[0] + 1
        if self.sub_sampling:
            self.time_kernel = self.add_weight(
                shape = (len(self.input_context), feature_dim, self.units),
                name = 'time_kernel',
                initializer = self.kernel_initializer,
                regularizer= self.kernel_regularizer,
                constraint=self.kernel_constraint)
        else:
            self.time_kernel = self.add_weight(
                shape = (context_size, feature_dim, self.units),
                name = 'time_kernel',
                initializer = self.kernel_initializer,
                regularizer= self.kernel_regularizer,
                constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(
                shape = (self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
            
        
    def call(self, inputs):
        
        if inputs.shape.rank != 3:  # [batch, time, feature]
          raise ValueError('inputs.shape.rank: %d must be 3 ' % inputs.shape.rank)
        if self.sub_sampling:
            dilations = self.input_context[-1] - self.input_context[0]
        else:
            dilations = 1
        output = tf.nn.conv1d(inputs,
                              self.time_kernel,
                              stride = self.stride, 
                              padding = self.pad,
                              dilations = dilations)
        if self.use_bias: 
            output = output + self.bias
        output = self.activation(output)
        return output
        
        
    def get_config(self):
      config = {
          'input_context': self.input_context,
          'sub_sampling': self.sub_sampling,
          'kernel_initializer': self.kernel_initializer,
          'kernel_regularizer': self.kernel_regularizer,
          'kernel_constraint': self.kernel_constraint,
          'bias_initializer': self.bias_initializer,
          'bias_regularizer': self.bias_regularizer,
          'bias_constraint': self.bias_constraint,
          'pad': self.pad,
      }
      base_config = super(TDNN, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        