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
import original_net as original
import time
import data_utils
import matplotlib.pyplot as plt
import PIL

class CycleGAN():

  def __init__(self, mode='train', base='unet', img_size=256, verbose=False):
    assert mode in ['train', 'test']
    assert base in ['unet', 'original']
    self.mode = mode
    self.base = unet if base == 'unet' else original
    self.img_shape = (img_size, img_size, 3)
    self.verbose = verbose

  def build(self):
    self.disc_a = self.build_discriminator('disc_a')
    self.disc_b = self.build_discriminator('disc_b')
    self.gen_a2b = self.build_generator('gen_a2b')
    self.gen_b2a = self.build_generator('gen_b2a')
    self.combined_a = self.build_combined(self.gen_a2b, self.gen_b2a, self.disc_b)
    self.combined_b = self.build_combined(self.gen_b2a, self.gen_a2b, self.disc_a)
    if self.verbose:
      self.disc_a.summary()
      self.disc_b.summary()
      self.gen_a2b.summary()
      self.gen_b2a.summary()
      self.combined_a.summary()
      self.combined_b.summary()

  def build_discriminator(self, name_base):
    img = KL.Input(self.img_shape)
    output = self.base.discriminator(img, name_base)
    return KM.Model(img, output)

  def build_generator(self, name_base):
    img = KL.Input(self.img_shape)
    output = self.base.generator(img, name_base)
    return KM.Model(img, output)
  
  def build_combined(self, gen1, gen2, disc):
    img = KL.Input(shape=self.img_shape)
    fake = gen1(img)
    fake_fake = gen2(fake)
    disc_out = disc(fake)
    return KM.Model(img, [disc_out, fake_fake])
  
  def compile(self, learning_rate):
    adam = keras.optimizers.Adam(learning_rate, 0.5)
    # n_disc_trainable = len(self.disc_a.trainable_weights)
    self.disc_a.compile(loss='mse', # keras.losses.binary_crossentropy,
      optimizer=adam, metrics=['accuracy'])
    self.disc_b.compile(loss='mse', # keras.losses.binary_crossentropy,
      optimizer=adam, metrics=['accuracy'])
    # don't update discrimators during the training of generators
    # for layer in self.disc_a.layers:
    #   layer.trainable = False
    # for layer in self.disc_b.layers:
    #   layer.trainable = False
    self.disc_a.trainable = False
    self.disc_b.trainable = False
    self.combined_a.compile(loss=['mse', 'mae'],
      loss_weights=[1, 10], optimizer=adam)
    self.combined_b.compile(loss=['mse', 'mae'],
      loss_weights=[1, 10], optimizer=adam)
    # print(n_disc_trainable, len(self.disc_a._collected_trainable_weights))
    # print(len(self.combined_a._collected_trainable_weights), len(self.gen_a2b.trainable_weights))
  
  def train(self, path_a, path_b, epochs=100, steps_per_epoch=1000, 
      batch_size=1, save_image_every_step=50, save_model_every_epoch=1,
      image_save_path='../output/images/', model_save_path='../models/',
      show_image=True, load_model=False, model_load_path='../models/'):
    print('Training starts...')
    # loading model
    if load_model:
      self.disc_a = KM.load_model(model_load_path + 'model-disc-a.h5')
      self.disc_b = KM.load_model(model_load_path + 'model-disc-b.h5')
      self.combined_a = KM.load_model(model_load_path + 'model-gan-a.h5')
      self.combined_b = KM.load_model(model_load_path + 'model-gan-b.h5')
      print('Loaded weights from', model_load_path)
    ones = np.ones((batch_size,) + self.base.DISC_OUTPUT_SIZE)
    zeros = np.zeros((batch_size,) + self.base.DISC_OUTPUT_SIZE)
    data_a = data_utils.get_train_dataset(path_a)
    data_b = data_utils.get_train_dataset(path_b)
    sess = tf.Session()
    for epoch in range(epochs):
      print('Epoch %d starts...' % epoch)
      for step in range(steps_per_epoch):
        start_time = time.time()
        # get image
        img_a = sess.run(data_a)
        img_b = sess.run(data_b)
        # generate images
        fake_b = self.gen_a2b.predict(img_a)
        fake_a = self.gen_b2a.predict(img_b)
        # fake_fake_a = self.gen_b2a.predict(fake_b)
        # fake_fake_b = self.gen_a2b.predict(fake_a)
        # train discriminators
        disc_a_loss_real = self.disc_a.train_on_batch(img_a, ones)
        disc_a_loss_fake = self.disc_a.train_on_batch(fake_a, zeros)
        disc_a_loss = 0.5 * np.add(disc_a_loss_fake, disc_a_loss_real)
        disc_b_loss_real = self.disc_b.train_on_batch(img_b, ones)
        disc_b_loss_fake = self.disc_b.train_on_batch(fake_b, zeros)
        disc_b_loss = 0.5 * np.add(disc_b_loss_fake, disc_b_loss_real)
        # train generators
        gen_a_loss = self.combined_a.train_on_batch(img_a, [ones, img_a])
        gen_b_loss = self.combined_b.train_on_batch(img_b, [ones, img_b])
        # plot
        print ("[Epoch %d/%d] [Batch %d/%d] [D_A loss: %f, acc: %3d%%] [D_B loss: %f, acc: %3d%%] [G_A loss: %f] [G_B loss: %f] time: %s" % (epoch+1, epochs,
                step+1, steps_per_epoch,
                disc_a_loss[0], 100*disc_a_loss[1],
                disc_b_loss[0], 100*disc_b_loss[1],
                gen_a_loss[0], gen_b_loss[0],
                time.time() - start_time))
        # save image
        if step % save_image_every_step == 0:
          if show_image:
            from IPython.display import clear_output
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
      # save model
      if epoch % save_model_every_epoch == 0:
        self.disc_a.save(model_save_path + 'model-disc-a.h5')
        self.disc_b.save(model_save_path + 'model-disc-b.h5')
        self.combined_a.save(model_save_path + 'model-gan-a.h5')
        self.combined_b.save(model_save_path + 'model-gan-b.h5')
        print('Saved model for epoch', epoch+1)

