import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib
matplotlib.use('Agg')
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For multiple devices (GPUs: 4, 5, 6, 7)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import math
import time, sys
import pickle
import timeit
import wandb
import shutil
import pandas as pd
import keras.backend as K 
from tensorflow.keras import layers
import keras
os.environ["WANDB_API_KEY"] = "a7b3bca989f1cf6c97c2cbf57f77de63403a5fe5"
plt.ioff() 
#from keras.constraints import Constraint
#class minmax_constraint(Constraint):
#    def __init__(self, min_value=-11.5, max_value=-2.25):
#        self.min_value = min_value
#        self.max_value = max_value
#
#    def __call__(self, w):        
#        return tf.clip_by_value(w, self.min_value, self.max_value)
#########
def x_Sigma_w_x_T(x, W_Sigma):
    batch_sz = x.shape[0]
    dim = tf.cast(tf.shape(x)[-1], tf.float32)
    x = x/dim
    xx_t = tf.reduce_sum(tf.multiply(x, x), axis=-1,
                         keepdims=True)  # [50, 17, 64]  -> [50, 17, 1] or [50, 64] - > [50, 1]
    # xx_t_e = tf.expand_dims(xx_t,axis=2)
    return tf.multiply(xx_t, W_Sigma)  # [50,17,64] or [50, 64] or [50, 10]
#########
def w_t_Sigma_i_w(w_mu, in_Sigma):  # [64, 64]  , [50, 17, 64] or [64, 10], [50, 64]
    dim = tf.math.sqrt(tf.cast(w_mu.shape[0], tf.float32))
    w_mu = w_mu/dim
    Sigma_1 = tf.matmul(in_Sigma, tf.multiply(w_mu, w_mu))  # [50, 17, 64] or [50, 10]
    return Sigma_1
#########
def tr_Sigma_w_Sigma_in(in_Sigma, W_Sigma):
    dim = tf.cast(W_Sigma.shape[-1], tf.float32)
    Sigma = tf.reduce_sum(in_Sigma, axis=-1, keepdims=True)  # [50,17, 1]
    return tf.multiply(Sigma, W_Sigma)/dim  # [50,17, 64]
#########
def activation_Sigma(gradi, Sigma_in):
    grad1 = tf.multiply(gradi, gradi)  # [50,17,64] or [50, 10]
    dim = tf.cast(grad1.shape[-1], tf.float32)
    return tf.multiply(Sigma_in, grad1)/dim  # [50,17,64] or [50, 10]
#########
def kl_regularizer_conv(mu, logvar): 
#    k = mu.shape[-1]
#    mu = tf.reshape(mu, [-1, k])
#    n= mu.shape[0]
    prior_var = 0.01
    kl_loss = tf.math.log(prior_var)  - 1 - logvar + (tf.math.log(1+tf.math.exp(logvar))/prior_var) + ( tf.square(mu)/prior_var)
    kl = 0.5*tf.math.reduce_mean( kl_loss)#/( tf.math.reduce_max( kl_loss) + tf.keras.backend.epsilon() )
   # kl = tf.where(tf.math.is_nan(kl), tf.constant(1.0e-5, shape=kl.shape), kl)
   # kl = tf.where(tf.math.is_inf(kl), tf.constant(1.0e-5, shape=kl.shape), kl)
    return kl
#########
def kl_regularizer(mu, logvar):
   # n= mu.shape[0]
    prior_var = 0.01
    kl_loss = tf.math.log(prior_var)  - 1 - logvar + (tf.math.log(1+tf.math.exp(logvar))/prior_var) + ( tf.square(mu)/prior_var)
    kl = 0.5*tf.math.reduce_mean( kl_loss)#/( tf.math.reduce_max( kl_loss) + tf.keras.backend.epsilon() )
    #kl = tf.where(tf.math.is_nan(kl), tf.constant(1.0e-5, shape=kl.shape), kl)
    #kl = tf.where(tf.math.is_inf(kl), tf.constant(1.0e-5, shape=kl.shape), kl)
    return kl 
#########
class VDP_first_Conv(keras.layers.Layer):
    def __init__(self, kernel_size=5, kernel_num=16, kernel_stride=1, padding="VALID",**kwargs):
        super(VDP_first_Conv, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.padding = padding
       # self.sigma_constraint = constraints.MaxNorm(max_value=1.0)
    def build(self, input_shape):
        # def sigma_regularizer_conv(x):
        #     f_s = tf.math.softplus(x)#  tf.math.log(1. + tf.math.exp(x))
        #     return (self.kernel_size* self.kernel_size*  input_shape[-1])*tf.reduce_mean(f_s-tf.math.log(f_s)-1.)
        ini_sigma = -2.25
        min_sigma = -4.6
        self.w_mu = self.add_weight(shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.kernel_num),
                                    initializer=tf.keras.initializers.GlorotNormal(),#tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None),
                                    name='w_mu',
                                    trainable=True,
                                    )
        self.w_sigma = self.add_weight(shape=(self.kernel_num,), initializer=tf.random_uniform_initializer(minval=min_sigma, maxval=ini_sigma,  seed=None), #tf.keras.initializers.GlorotUniform(),  
        name='w_sigma',  trainable=True, ) #constraint=minmax_constraint() #initializer=tf.constant_initializer(ini_sigma), 
    def call(self, mu_in):
        batch_size = mu_in.shape[0]
        num_channel = mu_in.shape[-1]
        kl_conv = kl_regularizer_conv(self.w_mu, self.w_sigma)
        w_sigma_2 =   tf.math.log(1+tf.math.exp(self.w_sigma) )
        mu_out = tf.nn.conv2d(mu_in, self.w_mu, strides=[1, self.kernel_stride, self.kernel_stride, 1],
                              padding=self.padding, data_format='NHWC')
        x_train_patches = tf.image.extract_patches(mu_in, sizes=[1, self.kernel_size, self.kernel_size, 1],
                                                   strides=[1, self.kernel_stride, self.kernel_stride, 1],
                                                   rates=[1, 1, 1, 1],
                                                   padding=self.padding)  # shape=[batch_size, image_size, image_size, kernel_size*kernel_size*num_channel]
        x_train_matrix = tf.reshape(x_train_patches, [batch_size, -1, self.kernel_size * self.kernel_size * num_channel])  # shape=[batch_size, image_size*image_size, patch_size*patch_size*num_channel]
        x_dim = tf.cast(tf.shape(x_train_matrix)[-1], tf.float32)
        x_train_matrix = tf.math.reduce_sum(tf.math.square(x_train_matrix)/x_dim, axis=-1) # shape=[batch_size, image_size*image_size] = [16, 576]
        X_XTranspose = tf.ones([1, 1, self.kernel_num]) * tf.expand_dims(x_train_matrix, axis=-1)
        Sigma_out = tf.multiply(w_sigma_2,  X_XTranspose)  # shape=[batch_size,image_size*image_size, kernel_num]
        Sigma_out = tf.reshape(Sigma_out, [batch_size,mu_out.shape[1], mu_out.shape[1] , self.kernel_num])
        Sigma_out = tf.math.softplus(Sigma_out)
        return mu_out, Sigma_out, kl_conv # mu_shape= [batch_size, image_size, image_size, kernel_num], sigma_shape=[batch_size, image_size, image_size, kernel_num]
##########
class VDP_intermediate_Conv(keras.layers.Layer):
    def __init__(self, kernel_size=5, kernel_num=16, kernel_stride=1, padding="VALID",**kwargs):
        super(VDP_intermediate_Conv, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.padding = padding
     #   self.sigma_constraint = constraints.MaxNorm(max_value=1.0)
    def build(self, input_shape):
        ini_sigma = -2.25
        min_sigma = -4.6
        self.w_mu = self.add_weight(shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.kernel_num),
                                    initializer=tf.keras.initializers.GlorotNormal(), #tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None), 
                                    name='w_mu',
                                    trainable=True, 
                                    )
        self.w_sigma = self.add_weight(shape=(self.kernel_num,),
                                       initializer=tf.random_uniform_initializer(minval=min_sigma, maxval=ini_sigma,  seed=None), #tf.keras.initializers.GlorotUniform(),#tf.constant_initializer(ini_sigma),  # , 
                                       name='w_sigma', trainable=True, 
                                       )#constraint=minmax_constraint()
                                    #   )
    def call(self, mu_in, Sigma_in): # [batch_size, image_size , image_size, channel]
        batch_size = mu_in.shape[0]
        num_channel = mu_in.shape[-1]  # shape=[batch_size, im_size, im_size, num_channel]
        kl_conv = kl_regularizer_conv(self.w_mu, self.w_sigma)
        w_sigma_2 =   tf.math.log(1+tf.math.exp(self.w_sigma) )
       # w_sigma_2 = tf.clip_by_value(t=w_sigma_2, clip_value_min=tf.constant(1e-6), clip_value_max=tf.constant(1e+6))
        mu_out = tf.nn.conv2d(mu_in, self.w_mu, strides=[1, self.kernel_stride, self.kernel_stride, 1],
                              padding=self.padding, data_format='NHWC')

        diag_sigma_patches = tf.image.extract_patches(Sigma_in, sizes=[1, self.kernel_size, self.kernel_size, 1],
                                                      strides=[1, self.kernel_stride, self.kernel_stride, 1],
                                                      rates=[1, 1, 1, 1], padding=self.padding) # shape=[batch_size, new_im_size, new_im_size, kernel_size*kernel_size*num_channel]

        diag_sigma_g = tf.reshape(diag_sigma_patches, [batch_size, -1, self.kernel_size * self.kernel_size * num_channel]) # shape=[batch_size, new_im_size*new_im_size,   self.kernel_size*self.kernel_size*num_channel ]
        mu_cov_square = tf.reshape(tf.math.multiply(self.w_mu, self.w_mu), [self.kernel_size * self.kernel_size * num_channel, self.kernel_num])  # shape[ kernel_size*kernel_size*num_channel,   kernel_num]
        mu_dim = tf.cast(tf.shape(mu_cov_square)[0], tf.float32)
        mu_wT_sigmags_mu_w = tf.matmul(diag_sigma_g, mu_cov_square/mu_dim)  # shape=[batch_size, new_im_size*new_im_size , kernel_num   ]
        trace = tf.math.reduce_sum(diag_sigma_g, 2, keepdims=True)  # shape=[batch_size,  new_im_size* new_im_size, 1]
        trace = tf.ones([1, 1, self.kernel_num]) * trace  # shape=[batch_size,  new_im_size*new_im_size, kernel_num]
        trace =tf.multiply(w_sigma_2, trace) /mu_dim # shape=[batch_size, , new_im_size*new_im_size, kernel_num]
        mu_in_patches = tf.reshape(tf.image.extract_patches(mu_in, sizes=[1, self.kernel_size, self.kernel_size, 1],
                                                            strides=[1, self.kernel_stride, self.kernel_stride, 1],
                                                            rates=[1, 1, 1, 1], padding=self.padding),
                                                           [batch_size, -1, self.kernel_size * self.kernel_size * num_channel])# shape=[batch_size, new_im_size*new_im_size, self.kernel_size*self.kernel_size*num_channel]
        mu_gT_mu_g = tf.math.reduce_sum(tf.math.multiply(mu_in_patches, mu_in_patches), axis=-1)  # shape=[batch_size, new_im_size*new_im_size]
        mu_gT_mu_g1 = tf.ones([1, 1,  self.kernel_num]) * tf.expand_dims(mu_gT_mu_g, axis=-1)     # shape=[batch_size, new_im_size*new_im_size, kernel_num]
        sigmaw_mu_gT_mu_g = tf.multiply(w_sigma_2, mu_gT_mu_g1) /mu_dim  # shape=[batch_size, new_im_size*new_im_size, kernel_num]
        Sigma_out = trace + mu_wT_sigmags_mu_w + sigmaw_mu_gT_mu_g  # # shape=[batch_size, new_im_size*new_im_size, kernel_num]
        Sigma_out = tf.reshape(Sigma_out, [batch_size,mu_out.shape[1], mu_out.shape[1] , self.kernel_num])
        Sigma_out = tf.math.softplus(Sigma_out)
        return mu_out, Sigma_out, kl_conv
###########
class VDP_MaxPooling(keras.layers.Layer):
    """VDP_MaxPooling"""
    def __init__(self, pooling_size=2, pooling_stride=2, pooling_pad='SAME',**kwargs):
        super(VDP_MaxPooling, self).__init__(**kwargs)
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.pooling_pad = pooling_pad

    def call(self, mu_in, Sigma_in): # shape=[batch_size,,im_size, im_size, num_channel]
        batch_size = mu_in.shape[0]  # shape=[batch_size, im_size, im_size, num_channel]
        hw_in = mu_in.shape[1]
        num_channel = mu_in.shape[-1]
        mu_out, argmax_out = tf.nn.max_pool_with_argmax(mu_in, ksize=[1, self.pooling_size, self.pooling_size, 1],
                                                        strides=[1, self.pooling_stride, self.pooling_stride, 1],
                                                        padding=self.pooling_pad)  # shape=[batch_zise, new_size,new_size,num_channel]
        hw_out = mu_out.shape[1]
        argmax1 = tf.transpose(argmax_out, [0, 3, 1, 2])
        argmax2 = tf.reshape(argmax1, [batch_size, num_channel,
                                       -1])  # shape=[batch_size, num_channel, new_size*new_size]
        x_index = tf.math.floormod(tf.compat.v1.floor_div(argmax2, tf.constant(num_channel,
                                                                               shape=[batch_size, num_channel,
                                                                                      hw_out * hw_out], dtype='int64')),
                                   tf.constant(hw_in, shape=[batch_size, num_channel, hw_out * hw_out], dtype='int64'))
        aux = tf.compat.v1.floor_div(tf.compat.v1.floor_div(argmax2, tf.constant(num_channel,
                                                                                 shape=[batch_size, num_channel,
                                                                                        hw_out * hw_out],  dtype='int64')),
                                     tf.constant(hw_in, shape=[batch_size, num_channel, hw_out * hw_out], dtype='int64'))
        y_index = tf.math.floormod(aux,  tf.constant(hw_in, shape=[batch_size, num_channel, hw_out * hw_out], dtype='int64'))
        index = tf.multiply(y_index, hw_in) + x_index  # shape=[batch_size, num_channel,new_size*new_size]
        Sigma_in1 = tf.transpose(tf.reshape(Sigma_in, [batch_size, -1, num_channel ]), [0, 2, 1])
        Sigma_out = tf.gather(Sigma_in1, index, batch_dims=2,     axis=-1)  # shape=[batch_size,num_channel,new_size*new_size]
        Sigma_out = tf.reshape(tf.transpose(Sigma_out,    [0, 2, 1]), [batch_size, mu_out.shape[1], mu_out.shape[1],num_channel])  # shape=[batch_size,new_size, new_size, num_channel]
        #Sigma_out = tf.math.softplus(Sigma_out)
        return mu_out, Sigma_out
####### 
class LinearFirst(keras.layers.Layer):
    """y = w.x + b"""
    def __init__(self, units,**kwargs):
        super(LinearFirst, self).__init__(**kwargs)
        self.units = units
      #  self.sigma_constraint = constraints.MaxNorm(max_value=1.0)
    def build(self, input_shape):
        ini_sigma = -2.25
        min_sigma = -4.6
        self.w_mu = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer=tf.keras.initializers.GlorotNormal(),#tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None), 
                                    name='w_mu',
                                    trainable=True)
        self.w_sigma = self.add_weight(shape=(self.units,),
                                    initializer=tf.random_uniform_initializer(minval=min_sigma, maxval=ini_sigma,  seed=None),#tf.keras.initializers.GlorotUniform(),#tf.constant_initializer(ini_sigma), 
                                       #constraint=minmax_constraint(), 
                                    name='w_sigma',
                                    trainable=True)
    def call(self, inputs):  # [50,17,64]
        # Mean
        # print(self.w_mu.shape)
        kl_fc = kl_regularizer(self.w_mu, self.w_sigma)
        mu_out = tf.matmul(inputs, self.w_mu)  # + self.b_mu       [50, 17, 64]             # Mean of the output
        # Varinace
        W_Sigma = tf.math.log(1+tf.math.exp(self.w_sigma) ) # [64]                        # Construct W_Sigma from w_sigmas
        #W_Sigma = tf.clip_by_value(t=W_Sigma, clip_value_min=tf.constant(1e-6), clip_value_max=tf.constant(1e+6))
        Sigma_out = x_Sigma_w_x_T(inputs, W_Sigma)  # [50, 17, 64]            + tf.math.log(1. + tf.math.exp(self.b_sigma)) #tf.linalg.diag(self.b_sigma)
        Sigma_out = tf.math.softplus(Sigma_out)
        return mu_out, Sigma_out, kl_fc

#######    
class LinearNotFirst(keras.layers.Layer):
    """y = w.x + b"""
    def __init__(self, units,**kwargs):
        super(LinearNotFirst, self).__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        ini_sigma = -2.25
        min_sigma = -4.6
        self.w_mu = self.add_weight(shape=(input_shape[-1], self.units),    # [64 , 64] or or [64, 10] or [10, 10]
                                    initializer=tf.keras.initializers.GlorotNormal(),#tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None),
                                    name='w_mu', 
                                    trainable=True, )
        self.w_sigma = self.add_weight(shape=(self.units,),
                                       initializer=tf.random_uniform_initializer(minval=min_sigma, maxval=ini_sigma,  seed=None),#tf.keras.initializers.GlorotUniform(),#tf.constant_initializer(ini_sigma), 
                                       #constraint=minmax_constraint(), 
                                       name='w_sigma',
                                       trainable=True, )
    def call(self, mu_in, Sigma_in):  # [50,17,64],  [50,17,64]   or [50, 64] or [50, 10]
        mu_out = tf.matmul(mu_in, self.w_mu)  # + self.b_mu  [50, 17, 64]
        kl_fc = kl_regularizer(self.w_mu, self.w_sigma)
        W_Sigma =  tf.math.log(1+ tf.math.exp(self.w_sigma)) # [64]
        Sigma_1 = w_t_Sigma_i_w(self.w_mu, Sigma_in)  # [50,17,64]
        Sigma_2 = x_Sigma_w_x_T(mu_in, W_Sigma)  # [50, 17, 64]
        Sigma_3 = tr_Sigma_w_Sigma_in(Sigma_in, W_Sigma)  # [50, 17, 64]
        Sigma_out = Sigma_1 + Sigma_2 + Sigma_3  # + tf.linalg.diag(tf.math.log(1. + tf.math.exp(self.b_sigma)))  #[50, 17, 64]
        Sigma_out = tf.math.softplus(Sigma_out)
        return mu_out, Sigma_out , kl_fc # mu_out=[50,17,64], Sigma_out = [50,17,64]

# Bayesian Activation Functions
class VDP_GeLU(keras.layers.Layer):
    def __init__(self,**kwargs):
        super(VDP_GeLU, self).__init__(**kwargs)
    def call(self, mu_in, Sigma_in):  # mu_in = [50,17,64], Sigma_in= [50,17,64]
        mu_out = tf.nn.gelu(mu_in)  # [50,17,64]
        with tf.GradientTape() as g:
            g.watch(mu_in)
            out = tf.nn.gelu(mu_in)
        gradi = g.gradient(out, mu_in)  # [50,17,64]
        Sigma_out = activation_Sigma(gradi, Sigma_in)
        Sigma_out = tf.math.softplus(Sigma_out)
        return mu_out, Sigma_out  # [50,2,17,64], [50,2,17,64,64]
    
class mysoftmax(keras.layers.Layer):
    def __init__(self,**kwargs):
        super(mysoftmax, self).__init__(**kwargs)
    def call(self, mu_in, Sigma_in):
        mu_dim = tf.cast(tf.shape(mu_in)[-1], tf.float32)
        mu_out = tf.nn.softmax(mu_in)
        grad = (mu_out - mu_out**2)**2
        Sigma_out = tf.multiply(grad, Sigma_in )/mu_dim
        Sigma_out = tf.math.softplus(Sigma_out)
        return mu_out, Sigma_out

class D_Dropout(keras.layers.Layer):
    def __init__(self, drop_prop,**kwargs):
        super(D_Dropout, self).__init__(**kwargs)
        self.drop_prop = drop_prop
    def call(self, mu_in, Training=True):
        # shape=[batch_size, seq length, embedding_dim]
        if Training:
            mu_out = tf.nn.dropout(mu_in, rate=self.drop_prop)  # [50,17,64] or [50, 10]
        else:
            mu_out = mu_in
        return mu_out 
        
class VDP_ReLU(keras.layers.Layer):
    """ReLU"""
    def __init__(self,**kwargs):
        super(VDP_ReLU, self).__init__(**kwargs)

    def call(self, mu_in, Sigma_in):
        mu_out = tf.nn.relu(mu_in)
        with tf.GradientTape() as g:
            g.watch(mu_in)
            out = tf.nn.relu(mu_in)
        gradi = g.gradient(out, mu_in)
        Sigma_out = activation_Sigma(gradi, Sigma_in)
        Sigma_out = tf.math.softplus(Sigma_out)
        return mu_out, Sigma_out

class VDP_Dropout(keras.layers.Layer):
    def __init__(self, drop_prop,**kwargs):
        super(VDP_Dropout, self).__init__(**kwargs)
        self.drop_prop = drop_prop

    def call(self, mu_in, Sigma_in, Training=True):
        # shape=[batch_size, seq length, embedding_dim]
        #scale_sigma = 1.0 / (1 - self.drop_prop)
        if Training:
            mu_out = tf.nn.dropout(mu_in, rate=self.drop_prop)  # [50,17,64] or [50, 10]
            # print('shape in dropout',mu_out.shape)
            non_zero = tf.not_equal(mu_out, tf.zeros_like(mu_out))  # [50,17,64]
            non_zero_sigma_mask = tf.boolean_mask(Sigma_in, non_zero)
            idx_sigma = tf.dtypes.cast(tf.where(non_zero), tf.int32)
            Sigma_out = tf.scatter_nd(idx_sigma, non_zero_sigma_mask, tf.shape(non_zero))/tf.cast(tf.shape(mu_out)[-1], tf.float32)
           # Sigma_out = (scale_sigma ** 2) * Sigma_out#tf.scatter_nd(idx_sigma, non_zero_sigma_mask, tf.shape(non_zero))/tf.cast(tf.shape(mu_out)[-1], tf.float32)
            # print('sigma shape in dropout',Sigma_out.shape)
        else:
            mu_out = mu_in
            Sigma_out = Sigma_in
       # Sigma_out = tf.math.softplus(Sigma_out)
        return mu_out, Sigma_out  # [50,17,64], [50,17,64]
    
# Bayesian Multi Layer Perceptron
class VDP_MLP(tf.keras.layers.Layer):
    def __init__(self, hidden_features, out_features, dropout_rate=0.1,**kwargs):
        super(VDP_MLP, self).__init__(**kwargs)
        self.dense1 = LinearNotFirst(hidden_features)
        self.dense2 = LinearNotFirst(out_features)
        self.dropout1 = VDP_Dropout(dropout_rate)
        self.gelu_1 = VDP_GeLU()
    def call(self, mu_in, sigma_in, training=True):
        mu_out, sigma_out, kl1 = self.dense1(mu_in, sigma_in)
        mu_out, sigma_out = self.gelu_1(mu_out, sigma_out)
        mu_out, sigma_out = self.dropout1(mu_out, sigma_out, Training = training)
        mu_out, sigma_out, kl2 = self.dense2(mu_out, sigma_out)
        mu_out, Sigma_out = self.dropout1(mu_out, sigma_out, Training = training)
        kl = kl1 + kl2
        Sigma_out = tf.math.softplus(Sigma_out)
        return mu_out, Sigma_out, kl

# Deterministic Layernorm
class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps 
        super(LayerNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=input_shape[-1:],
                                     initializer=tf.keras.initializers.Ones(), name='gamma',  trainable=True)
        self.beta = self.add_weight(shape=input_shape[-1:],
                                    initializer=tf.keras.initializers.Zeros(),name='beta',  trainable=True)
        super(LayerNorm, self).build(input_shape)
    def call(self, x):
        mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        std =tf.math.reduce_std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape


# Bayesian Layernorm
class Bayesian_LayerNorm(layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(Bayesian_LayerNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=input_shape[-1:],
                                     initializer=tf.keras.initializers.Ones(), name='gamma', trainable=True)
        self.beta = self.add_weight(shape=input_shape[-1:],
                                    initializer=tf.keras.initializers.Zeros(), name='beta', trainable=True)
        super(Bayesian_LayerNorm, self).build(input_shape)
    def call(self, mu_x, sigma_x):  # (batch_size, sequence_length, embedding_dim),  (batch_size, sequence_length, embedding_dim)[50,17, 64],[50,17,64]
        mean = tf.math.reduce_mean(mu_x, axis=-1, keepdims=True)  # [50,17,1]
        std = tf.math.reduce_std(mu_x, axis=-1, keepdims=True)  # [50,17,1]
        out_mu = self.gamma * (mu_x - mean) / (std + self.eps) + self.beta
      #  mean_sigma = tf.math.reduce_mean(sigma_x, axis=-1, keepdims=True)  # [50,17,1]
      #  std_sigma = tf.math.reduce_std(sigma_x, axis=-1, keepdims=True)  # [50,17,1]
        #Sigma_out = (self.gamma) * (sigma_x - mean_sigma) / (std_sigma + self.eps) #+ self.beta
        a = (self.gamma / (std + self.eps)) ** 2  # [50,17,64]
        Sigma_out = tf.math.multiply(a, sigma_x)  # [50,17,64]
        Sigma_out = tf.math.softplus(Sigma_out)
        return out_mu, Sigma_out
    def compute_output_shape(self, input_shape):
        return input_shape
    

# Bayesian Multi Head Attention
class Bayesian_MultiHeadSelfAttention_First(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.1,**kwargs):
        super(Bayesian_MultiHeadSelfAttention_First, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = LinearFirst(embed_dim)
        self.key_dense = LinearFirst(embed_dim)
        self.value_dense = LinearFirst(embed_dim)
        self.combine_heads = LinearNotFirst(embed_dim)
        self.dropout = D_Dropout(dropout)
        
    def build(self, input_shape):
        # No weights or parameters to initialize, so we pass
        pass    
    def attention(self, mu_query, sigma_query, mu_key, sigma_key, mu_value, sigma_value, input_dimension):
        dim_key = tf.cast(tf.shape(mu_key)[-1], tf.float32)
        mu_score = tf.matmul(mu_query, mu_key, transpose_b=True)  # [50, 2, 17, 32] x [50, 2, 32, 17] = [50, 2, 17, 17]
        # print('mu_score',mu_score.shape)
        a = tf.matmul((mu_query ** 2)/dim_key, sigma_key, transpose_b=True)  # [50, 2, 17, 17]
       # print('a', a.shape)
        b = tf.matmul((mu_key ** 2)/dim_key, sigma_query, transpose_b=True)  # [50, 2, 17, 17 ]
        c = tf.matmul(sigma_query/tf.math.sqrt(dim_key), sigma_key/tf.math.sqrt(dim_key), transpose_b=True) #[50, 2, 17, 17]
        sigma_score = a + b + c  # [50, 2, 17, 17]
        # print('sigma score',sigma_score.shape)
        
        mu_scaled_score = mu_score / tf.math.sqrt(dim_key)  # [50, 2, 17, 17]
        # print('mu scaled score',mu_scaled_score.shape)
        sigma_scaled_score = sigma_score / dim_key  # [50, 2, 17, 17]
        mu_weights = tf.nn.softmax(mu_scaled_score, axis=-1)  # [50, 2, 17, 17]
        # Sigma for softmax function
        weights_dim = tf.cast(tf.shape(mu_weights)[-1], tf.float32)
        grad = (mu_weights - mu_weights**2)**2
        Sigma_weights = tf.multiply(grad, sigma_scaled_score )/weights_dim
        mu_output = tf.matmul(mu_weights, mu_value)  # [50,2,17,17] X [50,2,17,32]=  [50,2,17,32]
        # print('mu output',mu_output.shape)       
        d = tf.matmul((mu_weights ** 2) /weights_dim , sigma_value)  # [50,2,17,32]
        e = tf.matmul(Sigma_weights, (mu_value ** 2)/weights_dim)  # [50,2,17,32]
        f = tf.matmul(Sigma_weights/tf.math.sqrt(weights_dim), sigma_value/tf.math.sqrt(weights_dim))  # [50, 2, 17, 17]x[50,2,17,32]=  [50,2,17,32]
        output_sigma = d + e + f
        output_sigma = tf.math.softplus(output_sigma)
        return mu_output, output_sigma  # , mu_weights, Sigma_weights

    def separate_heads(self, mu_x, sigma_x, batch_size):  # [50, 17,64], [50, 17, 64]
        mu_x = tf.reshape(mu_x, (batch_size, -1, self.num_heads, self.projection_dim))  # [50, 17, 2 ,32]
        # print('mu_x',mu_x.shape)
        sigma_x = tf.reshape(sigma_x, (batch_size, -1, self.num_heads, self.projection_dim))  # [50, 17, 2 32]
        mu_x = tf.transpose(mu_x, perm=[0, 2, 1, 3])  # [50, 2, 17, 32]
        sigma_x = tf.transpose(sigma_x, perm=[0, 2, 1, 3])  # [50, 2, 17, 32]
        return mu_x, sigma_x  # [50,2,17,32],[50,2,17,32]

    def call(self, inputs, training=True):
        batch_size = tf.shape(inputs)[0]
        inputs = self.dropout(inputs, Training=training) 
        mu_query, sigma_query, kl_q = self.query_dense(inputs)  # [50, 17,64]  , [50, 17,64]
        mu_key, sigma_key, kl_k = self.key_dense(inputs)  # [50, 17,64] , [50, 17,64]
        mu_value, sigma_value, kl_v = self.value_dense(inputs)  # [50, 17,64], [50, 17,64]

        mu_query, sigma_query = self.separate_heads(mu_query, sigma_query, batch_size)
        mu_key, sigma_key = self.separate_heads(mu_key, sigma_key, batch_size)
        mu_value, sigma_value = self.separate_heads(mu_value, sigma_value, batch_size)
        # print('query2 in MHA after passing through separate heads=',mu_query.shape)

        mu_attention, sigma_attention = self.attention(mu_query, sigma_query, mu_key, sigma_key, mu_value, sigma_value,
                                                       tf.shape(inputs)[1])
        mu_attention = tf.transpose(mu_attention, perm=[0, 2, 1, 3])  # [50,17,2,32]
        # print('mu attention',mu_attention.shape)
        sigma_attention = tf.transpose(sigma_attention, perm=[0, 2, 1, 3])
        # print('sigma attention',sigma_attention.shape)
        mu_concat_attention = tf.reshape(mu_attention, (batch_size, -1, self.embed_dim))
        # print("shape after concat_attention:",mu_concat_attention.shape) #[50,17,64]
        sigma_concat_attention = tf.reshape(sigma_attention, (batch_size, -1, self.embed_dim))
        # print("shape after concat_attention sigma:",sigma_concat_attention.shape) #[50,17,64]
        mu_output, Sigma_out, kl_comb = self.combine_heads(mu_concat_attention, sigma_concat_attention)
        kl = kl_q + kl_k + kl_v + kl_comb
        Sigma_out = tf.math.softplus(Sigma_out)
        return mu_output, Sigma_out, kl

class Bayesian_MultiHeadSelfAttention_Intermediate(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.1,**kwargs):
        super(Bayesian_MultiHeadSelfAttention_Intermediate, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = LinearNotFirst(embed_dim)
        self.key_dense = LinearNotFirst(embed_dim)
        self.value_dense = LinearNotFirst(embed_dim)
        self.combine_heads = LinearNotFirst(embed_dim)
        self.dropout1 = VDP_Dropout(dropout)
        
    def build(self, input_shape): 
        # No weights or parameters to initialize, so we pass
        pass        
    def attention(self, mu_query, sigma_query, mu_key, sigma_key, mu_value, sigma_value):
        dim_key = tf.cast(tf.shape(mu_key)[-1], tf.float32)
        mu_score = tf.matmul(mu_query, mu_key, transpose_b=True)  # [50, 2, 17, 32] x [50, 2, 32, 17] = [50, 2, 17, 17]
        # print('mu_score',mu_score.shape)
        a = tf.matmul((mu_query ** 2)/dim_key, sigma_key, transpose_b=True)  # [50, 2, 17, 17]
       # print('a', a.shape)
        b = tf.matmul((mu_key ** 2)/dim_key, sigma_query, transpose_b=True)  # [50, 2, 17, 17 ]
        c = tf.matmul(sigma_query/tf.math.sqrt(dim_key), sigma_key/tf.math.sqrt(dim_key), transpose_b=True) #[50, 2, 17, 17]
        sigma_score = a + b + c  # [50, 2, 17, 17]
        # print('sigma score',sigma_score.shape)
                
        mu_scaled_score = mu_score / tf.math.sqrt(dim_key)  # [50, 2, 17, 17]
        # print('mu scaled score',mu_scaled_score.shape)
        sigma_scaled_score = sigma_score / dim_key  # [50, 2, 17, 17]
        mu_weights = tf.nn.softmax(mu_scaled_score, axis=-1)  # [50, 2, 17, 17]
        weights_dim = tf.cast(tf.shape(mu_weights)[-1], tf.float32)
        grad = (mu_weights - mu_weights**2)**2
        Sigma_weights = tf.multiply(grad, sigma_scaled_score )/weights_dim
        mu_output = tf.matmul(mu_weights, mu_value)  # [50,2,17,17] X [50,2,17,32]=  [50,2,17,32]
        # print('mu output',mu_output.shape)
        #weights_dim = tf.cast(tf.shape(mu_weights)[-1], tf.float32)
        d = tf.matmul((mu_weights ** 2) /weights_dim , sigma_value)  # [50,2,17,32]
        e = tf.matmul(Sigma_weights, (mu_value ** 2)/weights_dim)  # [50,2,17,32]
        f = tf.matmul(Sigma_weights/tf.math.sqrt(weights_dim), sigma_value/tf.math.sqrt(weights_dim))  # [50, 2, 17, 17]x[50,2,17,32]=  [50,2,17,32]
        output_sigma = d + e + f
        output_sigma = tf.math.softplus(output_sigma)
        return mu_output, output_sigma  # , mu_weights, Sigma_weights

    def separate_heads(self, mu_x, sigma_x, batch_size):  # [50, 17,64], [50, 17, 64]
        mu_x = tf.reshape(mu_x, (batch_size, -1, self.num_heads, self.projection_dim))  # [50, 17, 2 ,32]
        # print('mu_x',mu_x.shape)
        sigma_x = tf.reshape(sigma_x, (batch_size, -1, self.num_heads, self.projection_dim))  # [50, 17, 2 32]
        mu_x = tf.transpose(mu_x, perm=[0, 2, 1, 3])  # [50, 2, 17, 32]
        sigma_x = tf.transpose(sigma_x, perm=[0, 2, 1, 3])  # [50, 2, 17, 32]
        return mu_x, sigma_x  # [50,2,17,32],[50,2,17,32]

    def call(self, mu_inputs, sigma_inputs, training=True):
        batch_size = tf.shape(mu_inputs)[0] 
        mu_inputs, sigma_inputs = self.dropout1(mu_inputs, sigma_inputs, Training=training)
        mu_query, sigma_query, kl_q = self.query_dense(mu_inputs, sigma_inputs)  # [50, 17,64]  , [50, 17,64]
        mu_key, sigma_key, kl_k = self.key_dense(mu_inputs, sigma_inputs)  # [50, 17,64] , [50, 17,64]
        mu_value, sigma_value, kl_v = self.value_dense(mu_inputs, sigma_inputs)  # [50, 17,64], [50, 17,64]

        mu_query, sigma_query = self.separate_heads(mu_query, sigma_query, batch_size)
        mu_key, sigma_key = self.separate_heads(mu_key, sigma_key, batch_size)
        mu_value, sigma_value = self.separate_heads(mu_value, sigma_value, batch_size)
        # print('query2 in MHA after passing through separate heads=',mu_query.shape)

        mu_attention, sigma_attention = self.attention(mu_query, sigma_query, mu_key, sigma_key, mu_value, sigma_value)
        mu_attention = tf.transpose(mu_attention, perm=[0, 2, 1, 3])  # [50,17,2,32]
        # print('mu attention',mu_attention.shape)
        sigma_attention = tf.transpose(sigma_attention, perm=[0, 2, 1, 3])
        # print('sigma attention',sigma_attention.shape)
        mu_concat_attention = tf.reshape(mu_attention, (batch_size, -1, self.embed_dim))
        # print("shape after concat_attention:",mu_concat_attention.shape) #[50,17,64]
        sigma_concat_attention = tf.reshape(sigma_attention, (batch_size, -1, self.embed_dim))
        # print("shape after concat_attention sigma:",sigma_concat_attention.shape) #[50,17,64]
        mu_output, Sigma_out, kl_comb = self.combine_heads(mu_concat_attention, sigma_concat_attention)
        kl = kl_q + kl_k + kl_v+ kl_comb
        Sigma_out = tf.math.softplus(Sigma_out)
        return mu_output, Sigma_out, kl

# Bayesian Transformer Block
class VDP_TransformerBlock_first(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1,**kwargs):
        super(VDP_TransformerBlock_first, self).__init__(**kwargs)
        self.att = Bayesian_MultiHeadSelfAttention_First(embed_dim, num_heads,dropout)  # [64,2]
        self.mlp = VDP_MLP(embed_dim * 2, embed_dim, dropout)  # [64*2,64,]
        self.layernorm1 = LayerNorm(eps=1e-6)
        self.layernorm2 = Bayesian_LayerNorm(eps=1e-6)
        self.dropout1 = VDP_Dropout(dropout)
        
    def build(self, input_shape):
        # No weights or parameters to initialize, so we pass
        pass
    def call(self, inputs, training=True):
        inputs_norm = self.layernorm1(inputs)  # [50,17,64]
        # print("output of first LN before MHA",inputs_norm.shape) #[50,17,64]
        mu_output, sigma_out, kl_att = self.att(inputs_norm,training=training)  # [50,17,64]
        # print("output of  MHA in TB",mu_output.shape)
        mu_output, sigma_out = self.dropout1(mu_output, sigma_out, Training=training)  # [50,17,64]
        # print("output of  MHA in TB after dropout",mu_output.shape)
        mu_out1 = mu_output + inputs  # [50,17,64]
        # print('output of of MHA before entering to MLP',mu_out1.shape)
        mu_out1_norm, sigma_out1_norm = self.layernorm2(mu_out1, sigma_out)
        mu_mlp_output, sigma_mlp_output, kl_mlp = self.mlp(mu_out1_norm, sigma_out1_norm, training = training)
       # mu_mlp_output, sigma_mlp_output = self.dropout1(mu_mlp_output, sigma_mlp_output, training=training)
        # print('2nd LN and MLP output',mu_mlp_output.shape)
        mu_output = mu_mlp_output + mu_out1
        Sigma_out = sigma_mlp_output +  sigma_out #tf.math.multiply(tf.math.multiply(gradi, gradi)/out_dim, sigma_out1)
        kl = kl_att + kl_mlp
        Sigma_out = tf.math.softplus(Sigma_out)
        return mu_output, Sigma_out, kl


class VDP_TransformerBlock_Intermediate(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.1,**kwargs):
        super(VDP_TransformerBlock_Intermediate, self).__init__(**kwargs)
        self.att = Bayesian_MultiHeadSelfAttention_Intermediate(embed_dim, num_heads,dropout)  # [64,2]
        self.mlp = VDP_MLP(embed_dim * 2, embed_dim, dropout)  # [64*2,64,dropout]
        self.layernorm1 = Bayesian_LayerNorm(eps=1e-6)
        self.layernorm2 = Bayesian_LayerNorm(eps=1e-6)
        self.dropout1 = VDP_Dropout(dropout)
    
    def build(self, input_shape):
        # No weights or parameters to initialize, so we pass
        pass

    def call(self, mu_inputs, sigma_inputs, training=True):
        mu_norm, sigma_norm = self.layernorm2(mu_inputs, sigma_inputs)  # [50,17,64]
        # print("output of first LN before MHA",inputs_norm.shape) #[50,17,64]
        mu_output, sigma_out, kl_att = self.att(mu_norm, sigma_norm, training=training)  # [50,17,64]
        # print("output of  MHA in TB",mu_output.shape)
        mu_output, sigma_out = self.dropout1(mu_output, sigma_out, Training=training)  # [50,17,64]
        # print("output of  MHA in TB after dropout",mu_output.shape)
        mu_out1 = mu_output + mu_inputs  # [50,17,64]
        sigma_out1 = sigma_out + sigma_inputs
        # print('output of of MHA before entering to MLP',mu_out1.shape)
        mu_out1_norm, sigma_out1_norm = self.layernorm2(mu_out1, sigma_out1)
        mu_mlp_output, sigma_mlp_output, kl_mlp = self.mlp(mu_out1_norm, sigma_out1_norm, training=training)
      #  mu_mlp_output, sigma_mlp_output = self.dropout1(mu_mlp_output, sigma_mlp_output, training=training)
        # print('2nd LN and MLP output',mu_mlp_output.shape)
        mu_output = mu_mlp_output + mu_out1
        Sigma_out = sigma_mlp_output + sigma_out1
        kl = kl_att + kl_mlp
        Sigma_out = tf.math.softplus(Sigma_out)
        return mu_output, Sigma_out, kl


class DDense(keras.layers.Layer):
    def __init__(self, units=32,**kwargs):
        '''
        Initialize the instance attributes
        '''
        super(DDense, self).__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        self.w = self.add_weight(name='kernel',
                                 shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
        )
        #self.b = self.add_weight(name='bias',
        #                         shape=(self.units,),
        #                         initializer=tf.zeros_initializer(),
        #                         )
    def call(self, inputs):
        return tf.matmul(inputs, self.w) #+ self.b
    

# Bayesian Vision Transformer
#@keras.saving.register_keras_serializable(package="VDP_ViT")
class VDP_ViT(tf.keras.Model):
    def __init__(
            self,
            image_size=28,
            patch_size=8,
            num_layers=2,
            num_classes=10,
            embed_dim=32,
            num_heads=4,
            mlp_dim=32,
            channels=1,
            dropout=0.3,
            **kwargs
    ):
        super(VDP_ViT, self).__init__(**kwargs)
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * (patch_size ** 2)
        self.image_size = image_size
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.channels=  channels
        self.dropout_rate = dropout
        self.pos_emb = self.add_weight(shape=(1, num_patches + 1, embed_dim),name = "pos_emb")
        self.class_emb = self.add_weight(shape=(1, 1, embed_dim),name = "class_emb")
        self.patch_proj = DDense(embed_dim)
        # self.enc_layers = VDP_TransformerBlock_first(d_model, num_heads, mlp_dim, dropout)  # for _ in range(num_layers)]
        # self.enc_layers =  [VDP_TransformerBlock(d_model, num_heads, mlp_dim, dropout)
        # for _ in range(num_layers) ]

        self.enc_layers1 = VDP_TransformerBlock_first(embed_dim, num_heads, dropout)
        self.enc_layers = [
            VDP_TransformerBlock_Intermediate(embed_dim, num_heads, dropout)
            for _ in range(self.num_layers)
        ]
        # self.mlp_head = VDP_MLP(mlp_dim, num_classes)
        self.mlp_head = VDP_MLP(mlp_dim, num_classes)
        self.softmax_out = mysoftmax()
        self.dropout = D_Dropout(dropout)
  #      self.dropout1 = VDP_Dropout(dropout)
  
    def build(self, input_shape):
        # This ensures the model is built before loading weights
        super().build(input_shape)
    
    def initialize_weights(self):
        """Initialize the model by running a forward pass with dummy data."""
        dummy_input = tf.random.normal((1, self.image_size, self.image_size, self.channels))
        _ = self(dummy_input, training=False)
        
    def load_weights_safely(self, weights_path):
        """Safely load weights after ensuring model is initialized."""
        # Initialize the model if not already done
        if not self.built:
            self.initialize_weights()
        # Load the weights
        self.load_weights(weights_path)       
        
    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches( 
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID")
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training=True):
       # print('Input dimension :', x.shape)
        batch_size = tf.shape(x)[0] 
        patches = self.extract_patches(x)
        # print('Input dimension after extract patch :',patches.shape)
        x = self.patch_proj(patches)
        # print('Input dimension after patch projection :',x.shape)
        class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.embed_dim])
        x = tf.concat([class_emb, x], axis=1)
        
        # print('Input dimension after concat :',x.shape)
        x = self.dropout(x, Training=training)
        x = x + self.pos_emb
        # mu_out, sigma_out = layer(x)
        mu_out, sigma_out, kl1 = self.enc_layers1(x,training=training)
        kl2 = 0
        for layer in self.enc_layers:
            mu_out, sigma_out, kl_2 = layer(mu_out, sigma_out, training=training)
            kl2 += kl_2
        # First (class token) is used for classification
        mu, sigma, kl_mlp = self.mlp_head(mu_out[:, 0], sigma_out[:, 0], training=training)
        
        mu, Sigma_out  =self.softmax_out(mu, sigma)  
       # print('shape of mu', mu.shape)
        # print('shape of sigma',sigma.shape) 
        kl = kl1 + kl2 + kl_mlp
        return mu, Sigma_out, kl