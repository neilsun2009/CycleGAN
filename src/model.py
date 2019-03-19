# Cycle GAN Model
# Using Keras with Tensorflow backend
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import unet
import resnet
import time
import data_utils
import matplotlib.pyplot as plt
import PIL
import losses

class CycleGAN():

  def __init__(self, mode='train', base='unet', img_size=256, verbose=False):
    assert mode in ['train', 'test']
    assert base in ['unet', 'resnet']
    self.mode = mode
    self.base = unet if base == 'unet' else resnet
    self.img_shape = (img_size, img_size, 3)
    self.verbose = verbose

  def build(self):
    self.disc_a = self.build_discriminator('disc_a')
    self.disc_b = self.build_discriminator('disc_b')
    self.disc_a_2 = self.build_discriminator_2('disc_a_2')
    self.disc_b_2 = self.build_discriminator_2('disc_b_2')
    self.gen_a2b = self.build_generator('gen_a2b')
    self.gen_b2a = self.build_generator('gen_b2a')
#     self.combined = self.build_combined(self.gen_a2b, self.gen_b2a, self.disc_a, self.disc_b)
    self.combined = self.build_combined(self.gen_a2b, self.gen_b2a, self.disc_a, self.disc_b, self.disc_a_2, self.disc_b_2)
    self.combined_disc_a = self.build_combined_disc(self.disc_a)
    self.combined_disc_b = self.build_combined_disc(self.disc_b)
    self.combined_disc_a_2 = self.build_combined_disc(self.disc_a_2)
    self.combined_disc_b_2 = self.build_combined_disc(self.disc_b_2)
    if self.verbose:
      self.disc_a.summary()
      self.disc_b.summary()
      self.disc_a_2.summary()
      self.disc_b_2.summary()
      self.gen_a2b.summary()
      self.gen_b2a.summary()
      self.combined.summary()

  def build_discriminator(self, name_base):
    img = KL.Input(self.img_shape)
    output = self.base.discriminator(img, name_base)
    return KM.Model(img, output)

  def build_discriminator_2(self, name_base):
    img = KL.Input(self.img_shape)
    output = self.base.discriminator_2(img, name_base)
    return KM.Model(img, output)

  def build_generator(self, name_base):
    img = KL.Input(self.img_shape)
    output = self.base.generator(img, name_base)
    return KM.Model(img, output)
  
  def build_combined(self, gen_a2b, gen_b2a, disc_a, disc_b, disc_a_2, disc_b_2):
    img_a = KL.Input(shape=self.img_shape)
    img_b = KL.Input(shape=self.img_shape)
    fake_b = gen_a2b(img_a)
    fake_a = gen_b2a(img_b)
    fake_fake_a = gen_b2a(fake_b)
    fake_fake_b = gen_a2b(fake_a)
    disc_out_a = disc_a(fake_a)
    disc_out_b = disc_b(fake_b)
    disc_out_a_2 = disc_a_2(fake_a)
    disc_out_b_2 = disc_b_2(fake_b)
    # identity mapping
    id_a = gen_b2a(img_a)
    id_b = gen_a2b(img_b)
    return KM.Model([img_a, img_b], [disc_out_a, disc_out_b, disc_out_a_2, disc_out_b_2, 
      fake_fake_a, fake_fake_b, id_a, id_b])

  def build_combined_disc(self, disc):
    img = KL.Input(shape=self.img_shape)
    fake = KL.Input(shape=self.img_shape)
    img_out = disc(img)
    fake_out = disc(fake)
    return KM.Model([img, fake], [img_out, fake_out])
  
  def compile(self, learning_rate, cycle_loss_weight=10, identity_loss_weight=1, disc_loss_weight=1, disc_2_loss_weight=0):
    adam_disc_a = keras.optimizers.Adam(learning_rate, 0.5)
    adam_disc_b = keras.optimizers.Adam(learning_rate, 0.5)
    adam_disc_a_2 = keras.optimizers.Adam(learning_rate, 0.5)
    adam_disc_b_2 = keras.optimizers.Adam(learning_rate, 0.5)
    adam_gen = keras.optimizers.Adam(learning_rate, 0.5)
    # n_disc_trainable = len(self.disc_a.trainable_weights)
    # self.disc_a.compile(loss=keras.losses.binary_crossentropy,
#     self.disc_a.compile(loss='mse', # keras.losses.binary_crossentropy,
#       optimizer=adam, metrics=['accuracy'])
#     # self.disc_b.compile(loss=keras.losses.binary_crossentropy,
#     self.disc_b.compile(loss='mse', # keras.losses.binary_crossentropy,
#       optimizer=adam, metrics=['accuracy'])
    self.combined_disc_a.compile(loss=[losses.mse, losses.mse], loss_weights=[0.5, 0.5], optimizer=adam_disc_a)
    self.combined_disc_b.compile(loss=[losses.mse, losses.mse], loss_weights=[0.5, 0.5], optimizer=adam_disc_b)
    self.combined_disc_a_2.compile(loss=[losses.mse, losses.mse], loss_weights=[0.5, 0.5], optimizer=adam_disc_a_2)
    self.combined_disc_b_2.compile(loss=[losses.mse, losses.mse], loss_weights=[0.5, 0.5], optimizer=adam_disc_b_2)
    # don't update discrimators during the training of generators
    # for layer in self.disc_a.layers:
    #   layer.trainable = False
    # for layer in self.disc_b.layers:
    #   layer.trainable = False
    self.disc_a.trainable = False
    self.disc_b.trainable = False
    self.disc_a_2.trainable = False
    self.disc_b_2.trainable = False
    # self.combined.compile(loss=[keras.losses.binary_crossentropy, keras.losses.binary_crossentropy, 'mae', 'mae', 'mae', 'mae'],
    self.combined.compile(loss=[losses.mse, losses.mse, losses.mse, losses.mse, losses.mae, losses.mae, losses.mae, losses.mae],
      loss_weights=[disc_loss_weight, disc_loss_weight, disc_2_loss_weight, disc_2_loss_weight, cycle_loss_weight, cycle_loss_weight, identity_loss_weight, identity_loss_weight],
      optimizer=adam_gen)
    # print(n_disc_trainable, len(self.disc_a._collected_trainable_weights))
    # print(len(self.combined._collected_trainable_weights), len(self.gen_a2b.trainable_weights))
#     print(self.combined._collected_trainable_weights)
  
  def test(self, img_path, model_path, is_a2b=True,
      batch_size=1, image_save_path='../output/images/', show_image=True, show_image_every_step=50):
    print('Testing starts...')
    if show_image:
      from IPython.display import clear_output
    # loading model
    self.combined.load_weights(model_path)
    print('Loaded weights from', model_path)
    # data
    data = data_utils.get_test_dataset(img_path, batch_size)
    sess = tf.Session()
    # iteration
    img_num = 0
    while True:
      start_time = time.time()
      try:
        # get image
        img = sess.run(data)
      except: 
        print('Test completed.')
        break
        # generate images
      if is_a2b:
        fake = self.gen_a2b.predict(img)
        fake_fake = self.gen_b2a.predict(fake)
      else:
        fake = self.gen_b2a.predict(img)
        fake_fake = self.gen_a2b.predict(fake)
      plt.figure(figsize=(15,15))
      titles = ['Input', 'Output', 'Reconstructed']
      displays = [img, fake, fake_fake]
      for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(titles[i])
        display = displays[i][0]
        plt.imshow(display * 0.5 + 0.5)
        plt.axis('off')
      plt.savefig('%s%d.jpg' % (image_save_path, img_num+1))
      if show_image and img_num % show_image_every_step == 0:
        clear_output(wait=True)
        plt.show()
      else:
        plt.close()
      print ("[Image %d] time: %s" % (img_num+1, time.time() - start_time))
      img_num += 1
      # end of test step
      

  def train(self, path_a, path_b, epochs=200, decay_from=100, steps_per_epoch=1000, 
      batch_size=1, true_label_value=1.0, save_image_every_step=50, save_model_every_epoch=1,
      image_save_path='../output/images/', model_save_path='../models/',
      show_image=True, load_model=False, model_load_path='../models/'):
    print('Training starts...')
    if show_image:
      from IPython.display import clear_output
    # loading model
    if load_model:
      self.combined_disc_a = KM.load_model(model_load_path + 'model-disc-a.h5')
      self.combined_disc_b = KM.load_model(model_load_path + 'model-disc-b.h5')
      self.combined_disc_a_2 = KM.load_model(model_load_path + 'model-disc-a-2.h5')
      self.combined_disc_b_2 = KM.load_model(model_load_path + 'model-disc-b-2.h5')
      self.combined = KM.load_model(model_load_path + 'model-gan.h5')
      print('Loaded weights from', model_load_path)
    ones = np.ones((batch_size,) + self.base.DISC_OUTPUT_SIZE) * true_label_value
    zeros = np.zeros((batch_size,) + self.base.DISC_OUTPUT_SIZE)
    ones_2 = np.ones((batch_size,) + self.base.DISC_2_OUTPUT_SIZE) * true_label_value
    zeros_2 = np.zeros((batch_size,) + self.base.DISC_2_OUTPUT_SIZE)
    data_a = data_utils.get_train_dataset(path_a, batch_size)
    data_b = data_utils.get_train_dataset(path_b, batch_size)
    sess = tf.Session()
    # calculate lr linear decay value
    lr_decay = K.get_value(self.combined.optimizer.lr) / (epochs - decay_from)
    # iteration
    for epoch in range(epochs):
      print('Epoch %d starts...' % (epoch+1))
      for step in range(steps_per_epoch):
        start_time = time.time()
        # get image
        img_a = sess.run(data_a)
        img_b = sess.run(data_b)
        # generate images
        fake_b = self.gen_a2b.predict(img_a)
        fake_a = self.gen_b2a.predict(img_b)
        # save image
        if step % save_image_every_step == 0:
          if show_image:
            clear_output(wait=True)
          plt.figure(figsize=(15,15))
          titles = ['Input A', 'Output A', 'Input B', 'Output B']
          displays = [img_a, fake_b, img_b, fake_a]
          for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.title(titles[i])
            display = displays[i][0]
            plt.imshow(display * 0.5 + 0.5)
            plt.axis('off')
          plt.savefig('%s%d-%d.jpg' % (image_save_path, epoch+1, step+1))
          if show_image:
            plt.show()
        # fake_fake_a = self.gen_b2a.predict(fake_b)
        # fake_fake_b = self.gen_a2b.predict(fake_a)
        # train generators
#         gen_loss = self.combined.train_on_batch([img_a, img_b], [ones, ones, 
#           img_a, img_b, img_a, img_b])
        gen_loss = self.combined.train_on_batch([img_a, img_b], [ones, ones, ones_2, ones_2, 
          img_a, img_b, img_a, img_b])
        # train discriminators
#         disc_a_loss_real = self.disc_a.train_on_batch(img_a, ones)
#         disc_a_loss_fake = self.disc_a.train_on_batch(fake_a, zeros)
#         disc_a_loss = 0.5 * np.add(disc_a_loss_fake, disc_a_loss_real)
#         disc_b_loss_real = self.disc_b.train_on_batch(img_b, ones)
#         disc_b_loss_fake = self.disc_b.train_on_batch(fake_b, zeros)
#         disc_b_loss = 0.5 * np.add(disc_b_loss_fake, disc_b_loss_real)
        disc_a_loss = self.combined_disc_a.train_on_batch([img_a, fake_a], [ones, zeros])
        disc_b_loss = self.combined_disc_b.train_on_batch([img_b, fake_b], [ones, zeros])
        disc_a_loss_2 = self.combined_disc_a_2.train_on_batch([img_a, fake_a], [ones_2, zeros_2])
        disc_b_loss_2 = self.combined_disc_b_2.train_on_batch([img_b, fake_b], [ones_2, zeros_2])
        # plot
        print ("[Epoch %d/%d] [Batch %d/%d] [D_A loss: %f] [D_B loss: %f] [D_A_2 loss: %f] [D_B_2 loss: %f] [G loss: %f] time: %s" % (epoch+1, epochs,
                step+1, steps_per_epoch,
                disc_a_loss[0],
                disc_b_loss[0],
                disc_a_loss_2[0],
                disc_b_loss_2[0],
                gen_loss[0], 
                time.time() - start_time))
        # end of training step
      # linear decay
      if (epoch >= decay_from):
#         for model in [self.combined_disc_a, self.combined_disc_b, self.combined]: 
        for model in [self.combined_disc_a, self.combined_disc_b, self.combined_disc_a_2, self.combined_disc_b_2, self.combined]: 
          new_lr = K.get_value(model.optimizer.lr) - lr_decay
          new_lr = max(new_lr, 0)
          K.set_value(model.optimizer.lr, new_lr)
        print('Learning rate decayed to', K.get_value(self.combined.optimizer.lr))
      # save model
      if epoch % save_model_every_epoch == 0:
        self.combined_disc_a.save(model_save_path + 'model-disc-a.h5')
        self.combined_disc_b.save(model_save_path + 'model-disc-b.h5')
        self.combined_disc_a_2.save(model_save_path + 'model-disc-a-2.h5')
        self.combined_disc_b_2.save(model_save_path + 'model-disc-b-2.h5')
        self.combined.save(model_save_path + 'model-gan.h5')
        print('Saved model for epoch', epoch+1)
