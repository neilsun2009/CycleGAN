from model import CycleGAN
import argparse

def test(imglist_a, imglist_b, model_path, base_net='resnet', batch_size=1, image_save_path_a='../output/cat_2_dog_test_cl3/testA/',
          image_save_path_b='../output/cat_2_dog_test_cl3/testB/', show_image_every_step=50, show_image=False):

  gan = CycleGAN(mode='test', base=base_net, verbose=False)
  gan.build()
  # gan.compile(learning_rate=lr, cycle_loss_weight=cycle_loss_weight, identity_loss_weight=identity_loss_weight, 
  #             disc_loss_weight=disc_loss_weight, disc_2_loss_weight=disc_2_loss_weight)
  gan.test(imglist_a, model_path, is_a2b=True,
         batch_size=1, image_save_path=image_save_path_a, show_image=show_image, show_image_every_step=show_image_every_step)
  gan.test(imglist_b, model_path, is_a2b=False,
         batch_size=1, image_save_path=image_save_path_b, show_image=show_image, show_image_every_step=show_image_every_step)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--imglist_a', default='../dataset/cat2dog_clean/test_set/dogs/*.jpg', help='image list a path')
  parser.add_argument('--imglist_b', default='../dataset/cat2dog_clean/test_set/cats/*.jpg', help='image list b path')
  parser.add_argument('--model_path', default='../models/cat_2_dog_cl3/model-gan.h5', help='path for loading the model')  
  parser.add_argument('--base_net', default='resnet',
    choices=['resnet', 'unet'], help="base net type")
  parser.add_argument('--batch_size', default=1, type=int, help='batch size')
  parser.add_argument('--show_image', default=False, type=bool, help='whether show image using matplotlib')
  parser.add_argument('--show_image_every_step', default=50, type=int, help='number of steps to show an image')  
  parser.add_argument('--image_save_path_a', default='../output/cat_2_dog_test_cl3/testA/', help='path for saving the output images from list a')
  parser.add_argument('--image_save_path_b', default='../output/cat_2_dog_test_cl3/testB/', help='path for saving the output images from list b')
  config, _ = parser.parse_known_args()
  test(**vars(config))

if __name__ == '__main__':
  main()

