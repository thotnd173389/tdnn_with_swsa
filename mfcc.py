#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 20:31:41 2020

@author: thorius
"""
import tensorflow as tf

class MFCC(tf.keras.layers.Layer):
    def __init__(self,
                 sample_rate = 16000.0,
                 dct_num_features = 40,
                 frame_size_ms = 25.0,
                 frame_step_ms = 10.0,
                 pad_end = True,
                 mel_lower_edge_hertz=20.0,
                 mel_upper_edge_hertz=7600.0,
                 mel_num_bins = 80,
                 log_epsilon=1e-12,
                 **kwargs):
        super(MFCC, self).__init__(**kwargs)
        
        self.sample_rate = sample_rate
        self.dct_num_features = dct_num_features
        self.frame_size_ms = frame_size_ms
        self.frame_step_ms = frame_step_ms
        self.pad_end = pad_end
        self.mel_lower_edge_hertz = mel_lower_edge_hertz
        self.mel_upper_edge_hertz = mel_upper_edge_hertz
        self.mel_num_bins = mel_num_bins
        self.log_epsilon = log_epsilon
            
        # convert milliseconds to discrete samples
        self.frame_size = int(round(sample_rate * frame_size_ms / 1000.0))
        self.frame_step = int(round(sample_rate * frame_step_ms / 1000.0))
        
        
        
    def build(self, input_shape):
        super(MFCC, self).build(input_shape)
        
    def call(self, inputs):
        super(MFCC, self).call(inputs)
        stfts = tf.signal.stft(inputs, 
                               frame_length = self.frame_size, 
                               frame_step = self.frame_step,
                               fft_length = 1024,
                               pad_end=self.pad_end)
        
        spectrograms = tf.abs(stfts[:,:-1,:])
        print("spectrograms: ", spectrograms.shape)
        
        num_spectrogram_bins = stfts.shape[-1]
        print("num_spectrogram_bins: ", num_spectrogram_bins)
        
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                                  self.mel_num_bins, 
                                  num_spectrogram_bins, 
                                  self.sample_rate, 
                                  self.mel_lower_edge_hertz,
                                  self.mel_upper_edge_hertz)
        
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
          linear_to_mel_weight_matrix.shape[-1:]))
        
        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + self.log_epsilon)
        print("log_mel_spectrogram shape", log_mel_spectrograms.shape)
        
        # Compute MFCCs from log_mel_spectrograms 
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :40]
        
        return mfccs
        