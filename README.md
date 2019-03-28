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

## Sample models & outputs
In folder ```samples```.
+ cat_2_dog_cl3: baseline model for cat2dog, cycle loss weight=3|disc1 loss weight=0.5|disc2 loss weight=0.5, note a2b is dog2cat, b2a is cat2dog
+ cat_2_dog_cl5: comparing model for cat2dog, cycle loss weight=5|disc1 loss weight=0.5|disc2 loss weight=0.5, note a2b is dog2cat, b2a is cat2dog
+ man2woman: test model for network structure, using celebrit dataset, cycle loss weight=10|disc1 loss weight=1|disc2 loss weight=0, only partially trained for 60 epochs, note a2b is man2woman, b2a is woman2man
