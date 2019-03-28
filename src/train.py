from model import CycleGAN
import argparse

def train(imglist_a, imglist_b, base_net='resnet', lr=2e-4, cycle_loss_weight=3, identity_loss_weight=0, 
          disc_loss_weight=0.5, disc_2_loss_weight=0.5, epochs=200, decay_from=100, 
          steps_per_epoch=3000, true_label_value=1, batch_size=1, image_save_path='../output/20190325/', model_save_path='../models/20190325/',
          save_image_every_step=100, save_model_every_epoch=1,
          show_image=False, load_model=False, model_load_path='../models/20190325/'):

  gan = CycleGAN(mode='train', base=base_net, verbose=False)
  gan.build()
  gan.compile(learning_rate=lr, cycle_loss_weight=cycle_loss_weight, identity_loss_weight=identity_loss_weight, 
              disc_loss_weight=disc_loss_weight, disc_2_loss_weight=disc_2_loss_weight)
  gan.train(imglist_a, imglist_b, epochs=epochs, decay_from=decay_from, 
            steps_per_epoch=steps_per_epoch, true_label_value=true_label_value, 
            batch_size=batch_size, image_save_path=image_save_path, model_save_path=model_save_path,
            save_image_every_step=save_image_every_step, save_model_every_epoch=save_model_every_epoch,
            show_image=show_image, load_model=load_model, model_load_path=model_load_path)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--imglist_a', default='../dataset/cat2dog_clean/training_set/dogs/*.jpg',
    help='image list a path')
  parser.add_argument('--imglist_b', default='../dataset/cat2dog_clean/training_set/cats/*.jpg',
    help='image list b path')
  parser.add_argument('--base_net', default='resnet',
    choices=['resnet', 'unet'], help="base net type")
  parser.add_argument('--lr', default=2e-4, type=float, help="learning rate")
  parser.add_argument('--cycle_loss_weight', default=3, type=float, help='cycle loss weight')
  parser.add_argument('--identity_loss_weight', default=0, type=float, help='identity loss weight')
  parser.add_argument('--disc_loss_weight', default=0.5, type=float, help='discriminator 1 loss weight')
  parser.add_argument('--disc_2_loss_weight', default=0.5, type=float, help='discriminator 2 loss weight')
  parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
  parser.add_argument('--decay_from', default=100, type=int, help='epoch number from which to perform lr linear decay')
  parser.add_argument('--steps_per_epoch', default=3000, type=int, help='number of steps per epoch')
  parser.add_argument('--true_label_value', default=1, type=float, help='value for true label')
  parser.add_argument('--batch_size', default=1, type=int, help='batch size')
  parser.add_argument('--save_image_every_step', default=100, type=int, help='number of steps to save an image')
  parser.add_argument('--save_model_every_epoch', default=1, type=int, help='number of epochs to save the model')
  parser.add_argument('--show_image', default=False, type=bool, help='whether show image using matplotlib')
  parser.add_argument('--load_model', default=False, type=bool, help='whether load a model to continue training')
  parser.add_argument('--image_save_path', default='../output/20190325/', help='path for saving the images')
  parser.add_argument('--model_save_path', default='../models/20190325/', help='path for saving the model')
  parser.add_argument('--model_load_path', default='../models/20190325/', help='path for loading the model')
  config, _ = parser.parse_known_args()
  train(**vars(config))

if __name__ == '__main__':
  main()

