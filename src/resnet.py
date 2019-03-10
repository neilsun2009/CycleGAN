# ResNet base model
# Using ResNet as generator transformation phase, Patch GAN as discriminator

import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

DISC_OUTPUT_SIZE = (16, 16, 1)

def downsample(x, filters, size, strides, name, apply_batchnorm=True): # for both gen and disc
  initializer = keras.initializers.RandomNormal(0, 0.02)
  output = KL.Conv2D(filters, kernel_size=size, name=name,
                      strides=strides, padding='same',
                      kernel_initializer=initializer)(x)
  if apply_batchnorm:
    output = KL.BatchNormalization()(output) # always use training mode
  output = KL.LeakyReLU()(output)
  return output

def res_block(x, filters, name):
  out = downsample(x, filters, 3, 1, name=name+'_1')
  out = downsample(out, filters, 3, 1, name=name+'_2')
  out = KL.Add()([out, x])
  out = KL.ReLU()(out)
  return out

def gen_upsample(x, filters, size, name, apply_dropout=False):
  initializer = keras.initializers.RandomNormal(0, 0.02)
  output = KL.Conv2DTranspose(filters, kernel_size=size,
                           strides=2, padding='same', name=name,
                           kernel_initializer=initializer)(x)
  output = KL.BatchNormalization()(output)
  if apply_dropout:
    output = KL.Dropout(rate=0.5)(output) # always use training mode
  output = KL.ReLU()(output)
  return output

# Patch GAN
def discriminator(x, name_base):
  initializer = keras.initializers.RandomNormal(0, 0.02)
  # like encode
  # output = KL.Concatenate(axis=-1)([x, y]) # 256
  output = downsample(x, 64, 4, 2, name_base + '_a', apply_batchnorm=False) # 128
  output = downsample(output, 128, 4, 2, name_base + '_b') # 64
  output = downsample(output, 256, 4, 2, name_base + '_c') # 32
  output = downsample(output, 512, 4, 2, name_base + '_d') # 16
  output = KL.Conv2D(1, kernel_size=4, # activation="sigmoid",
                            strides=1, padding='same',
                            kernel_initializer=initializer)(output) # 16
  # print(output)
  return output

# Generator with ResNet as transformation
def generator(x, name_base): # 256
  initializer = keras.initializers.RandomNormal(0, 0.02)
  # encode
  x1 = downsample(x, 64, 7, 1, name=name_base + '_down_a') # 256
  x2 = downsample(x1, 128, 3, 2, name=name_base + '_down_b') # 128
  x3 = downsample(x2, 256, 3, 2, name=name_base + '_down_c') # 64
  # transform
  t1 = res_block(x3, 256, name=name_base + '_res_a') # 64
  t2 = res_block(t1, 256, name=name_base + '_res_b') # 64
  t3 = res_block(t2, 256, name=name_base + '_res_c') # 64
  t4 = res_block(t3, 256, name=name_base + '_res_d') # 64
  t5 = res_block(t4, 256, name=name_base + '_res_e') # 64
  t6 = res_block(t5, 256, name=name_base + '_res_f') # 64
  t7 = res_block(t6, 256, name=name_base + '_res_f') # 64
  t8 = res_block(t7, 256, name=name_base + '_res_f') # 64
  t9 = res_block(t8, 256, name=name_base + '_res_f') # 64
  # decode
  x4 = gen_upsample(t9, 128, 3, name=name_base + '_up_a') # 128
  x5 = gen_upsample(x4, 64, 3, name=name_base + '_up_b') # 256
  output = KL.Conv2D(3, kernel_size=7,
                      strides=1, padding='same', name=name_base + '_up_c',
                      kernel_initializer=initializer,
                      activation='tanh')(x5)
  return output

