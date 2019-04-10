from model import CycleGAN
import argparse
import video2jpg

def test( imglist_b, model_path, base_net='resnet', batch_size=1, image_save_path_b='../output/cat_2_dog_test_cl3/testB/',
          show_image_every_step=50, show_image=False, plot_recon=False):

  gan = CycleGAN(mode='test', base=base_net, verbose=False)
  gan.build()
  # gan.compile(learning_rate=lr, cycle_loss_weight=cycle_loss_weight, identity_loss_weight=identity_loss_weight, 
  #             disc_loss_weight=disc_loss_weight, disc_2_loss_weight=disc_2_loss_weight)
  gan.test(imglist_b, model_path, is_a2b=False,
         batch_size=1, image_save_path=image_save_path_b, show_image=show_image, show_image_every_step=show_image_every_step, plot_recon=plot_recon)

def main():
  video2jpg.v2j('../dataset/video/cat1.avi', '../output/video2jpg_test/video2jpg/')
  parser = argparse.ArgumentParser()
  parser.add_argument('--imglist_b', default='../output/video2jpg_test/video2jpg/*.jpg', help='image list b path')
  parser.add_argument('--model_path', default='../models/cat_2_dog_cl3/model-gan.h5', help='path for loading the model')  
  parser.add_argument('--base_net', default='resnet',
    choices=['resnet', 'unet'], help="base net type")
  parser.add_argument('--batch_size', default=1, type=int, help='batch size')
  parser.add_argument('--show_image', default=False, type=bool, help='whether show image using matplotlib')
  parser.add_argument('--show_image_every_step', default=50, type=int, help='number of steps to show an image')  
  parser.add_argument('--image_save_path_b', default='../output/video2jpg_test/c&d_transform/', help='path for saving the output images from list b')
  parser.add_argument('--plot_recon', default=False, type=bool, help='format of the output')
  config, _ = parser.parse_known_args()
  test(**vars(config))
  video2jpg.j2v('../output/video2jpg_test/c&d_transform/', '../output/video2jpg_test/jpg2video/result.avi')

if __name__ == '__main__':
  main()

