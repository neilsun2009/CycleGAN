# U Net base model
# Using U Net as generator, Patch GAN as discriminator

import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

DISC_OUTPUT_SIZE = (30, 30, 1)

def downsample(x, filters, size, name, apply_batchnorm=True): # for both gen and disc
  initializer = keras.initializers.RandomNormal(0, 0.02)
  output = KL.Conv2D(filters, kernel_size=size, name=name,
                      strides=2, padding='same', use_bias=False,
                      kernel_initializer=initializer)(x)
  if apply_batchnorm:
    output = KL.BatchNormalization()(output) # always use training mode
  output = KL.LeakyReLU()(output)
  return output

def gen_upsample(x, x2, filters, size, name, apply_dropout=False):
  initializer = keras.initializers.RandomNormal(0, 0.02)
  output = KL.Conv2DTranspose(filters, kernel_size=size,
                           strides=2, padding='same', use_bias=False, name=name,
                           kernel_initializer=initializer)(x)
  output = KL.BatchNormalization()(output)
  if apply_dropout:
    output = KL.Dropout(rate=0.5)(output) # always use training mode
  output = KL.ReLU()(output)
  output = KL.Concatenate(axis=-1)([output, x2]) # skip connection
  return output

# Patch GAN
def discriminator(x, name_base):
  initializer = keras.initializers.RandomNormal(0, 0.02)
  # like encode
  # output = KL.Concatenate(axis=-1)([x, y]) # 256
  output = downsample(x, 64, 4, name_base + '_a', apply_batchnorm=False) # 128
  output = downsample(output, 128, 4, name_base + '_b') # 64
  output = downsample(output, 256, 4, name_base + '_c') # 32
  # we are zero padding here with 1 because we need our shape to 
  # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
  # paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
  # output = tf.pad(output, paddings) # 34
  output = KL.ZeroPadding2D(padding=((1, 1), (1, 1)))(output)
  output = KL.Conv2D(512, kernel_size=4,
                            strides=1, padding='valid', use_bias=False,
                            kernel_initializer=initializer)(output) # 31
  output = KL.BatchNormalization()(output)
  output = KL.LeakyReLU()(output)
  # padding same way
  # output = tf.pad(output, paddings) # 33
  output = KL.ZeroPadding2D(padding=((1, 1), (1, 1)))(output)
  output = KL.Conv2D(1, kernel_size=4,
                            strides=1, padding='valid',
                            kernel_initializer=initializer)(output) # 30
  # don't add a sigmoid activation here since
  # the loss function expects raw logits.
  # print(output)
  return output

# U-Net
def generator(x, name_base): # 256
  initializer = keras.initializers.RandomNormal(0, 0.02)
  # encode
  x1 = downsample(x, 64, 4, name=name_base + '_down_a', apply_batchnorm=False) # 128
  x2 = downsample(x1, 128, 4, name=name_base + '_down_b') # 64
  x3 = downsample(x2, 256, 4, name=name_base + '_down_c') # 32
  x4 = downsample(x3, 512, 4, name=name_base + '_down_d') # 16
  x5 = downsample(x4, 512, 4, name=name_base + '_down_e') # 8
  x6 = downsample(x5, 512, 4, name=name_base + '_down_f') # 4
  x7 = downsample(x6, 512, 4, name=name_base + '_down_g') # 2
  x8 = downsample(x7, 512, 4, name=name_base + '_down_h') # 1
  # decode
  x9 = gen_upsample(x8, x7, 512, 4, name=name_base + '_up_a', apply_dropout=True) # 2
  x10 = gen_upsample(x9, x6, 512, 4, name=name_base + '_up_b', apply_dropout=True) # 4
  x11 = gen_upsample(x10, x5, 512, 4, name=name_base + '_up_c', apply_dropout=True) # 8
  x12 = gen_upsample(x11, x4, 512, 4, name=name_base + '_up_d') # 16
  x13 = gen_upsample(x12, x3, 256, 4, name=name_base + '_up_e') # 32
  x14 = gen_upsample(x13, x2, 128, 4, name=name_base + '_up_f') # 64
  x15 = gen_upsample(x14, x1, 64, 4, name=name_base + '_up_g') # 128
  output = KL.Conv2DTranspose(3, kernel_size=4,
                          strides=2, padding='same', name=name_base + '_up_h',
                          kernel_initializer=initializer,
                          activation='tanh')(x15)
  return output

