#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:24:04 2020

@author: thorius
"""
import tensorflow as tf
from attend import Attend

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
                 mode = 'scaled-dot',
                 num_head = 1,
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
        
        self.mode = mode
        self.num_head = num_head
        
    def build(self, input_shape):
        super(SWSA, self).build(input_shape)
        self.attends = []
        if self.mode == 'scaled-dot':
            self.attend = Attend(self.units, 
                                 self.weights_initializer,
                                 self.weights_regularizer,
                                 self.weights_constraint,
                                 self.bias_initializer,
                                 self.bias_regularizer,
                                 self.bias_constraint,
                                 self.activation)
        elif self.mode == 'multi-head':
            for i in range(self.num_head):
                self.attends.append(Attend(int(self.units/self.num_head),
                                            self.weights_initializer,
                                            self.weights_regularizer,
                                            self.weights_constraint,
                                            self.bias_initializer,
                                            self.bias_regularizer,
                                            self.bias_constraint,
                                            self.activation))
        else:
            raise ValueError('mode argument must be scaled-dot or multi-head!!!')
            
        
        
        
    def call(self, inputs):
        
        if inputs.shape.rank != 3:  
            raise ValueError('inputs.shape.rank: %d must be 3 ' % inputs.shape.rank)
        
        if self.mode == 'scaled-dot':
            output = self.attend(inputs)
        elif self.mode == 'multi-head':
            output = tf.keras.layers.concatenate([attend(inputs) for attend in self.attends])
        return output
        
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