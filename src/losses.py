# loss functions
import tensorflow as tf


def mae(y_true, y_pred):
  return tf.reduce_mean(tf.abs(y_pred - y_true))

def mse(y_true, y_pred):
  return tf.reduce_mean(tf.squared_difference(y_pred, y_true))