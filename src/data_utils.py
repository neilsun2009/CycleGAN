# data related utility functions

import numpy as np
import tensorflow as tf

def get_train_dataset(path, batch_size=1, buffer_size=1000):
  train_dataset = tf.data.Dataset.list_files(path)
  train_dataset = train_dataset.shuffle(buffer_size)
  train_dataset = train_dataset.map(lambda x: load_image(x)).repeat()
  train_dataset = train_dataset.batch(batch_size)
  return train_dataset.make_one_shot_iterator().get_next()

def load_image(image_file, img_size=256, mode='train'):
  # print(image_file)
  assert mode in ['train', 'test']
  input_image = tf.read_file(image_file)
  input_image = tf.image.decode_jpeg(input_image, channels=3)
  input_image = tf.cast(input_image, tf.float32) # RGB [0, 255]
  if mode == 'train':
    # random jittering
    # resize to 286 * 286 * 3
    input_image = tf.image.resize_images(input_image, [286, 286],
                                       align_corners=True,
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # random cropping to 256 * 256 * 3
    input_image = tf.random_crop(input_image, size=[img_size, img_size, 3])
    input_image = (input_image / 127.5) - 1
    # print(input_image.shape) 
    return input_image
    
#     if np.random.random() > 0.5:
#       input_image = tf.image.flip_left_right(input_image)
#       real_image = tf.image.flip_left_right(real_image)
  else:
    input_image = tf.image.resize_images(input_image, size=[img_size, img_size], 
                                         align_corners=True, method=tf.image.ResizeMethod.BILINEAR)
    # normalize to [-1, 1]
    input_image = (input_image / 127.5) - 1
    return input_image