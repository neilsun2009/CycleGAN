[![996.ICU](http://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)
# CycleGAN
a Cycle GAN implementation using Keras with Tensorflow backend

## Clone
Use git lfs to retrieve large files (dataset & model h5 files).
```
git lfs clone git@github.com:neilsun2009/CycleGAN.git
```

## Train
Follow the parameter descriptions in ```src/train.py```

## Test
Follow the parameter descriptions in ```src/test.py```

## Source code description
+ cycle_gan.ipynb: a jupyter notebook for training and testing
+ data_utils.py: data loader & preprocessor
+ losses.py: loss functions
+ model.py: model class, including building, compiling, training & testing
+ resnet.py: network structure using ResNet as proposed in the original paper
+ unet.py: network structure using U-Net as proposed in the pix2pix paper
+ train.py: standalone training script
+ test.py: standalone testing script
+ video2jpg.py: video and image conversion
+ videotest.py: video conversion testing script

## Sample models & outputs
In folder ```samples```.
+ cat_2_dog_cl3: baseline model for cat2dog, cycle loss weight=3|disc1 loss weight=0.5|disc2 loss weight=0.5, note a2b is dog2cat, b2a is cat2dog
+ cat_2_dog_cl5: comparing model for cat2dog, cycle loss weight=5|disc1 loss weight=0.5|disc2 loss weight=0.5, note a2b is dog2cat, b2a is cat2dog
+ man2woman: test model for network structure, using celebrit dataset, cycle loss weight=10|disc1 loss weight=1|disc2 loss weight=0, only partially trained for 60 epochs, note a2b is man2woman, b2a is woman2man

