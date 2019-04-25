# TSimaging Vessel Segmentation
## Intern student project in TSimaing (independent working)

We are trying to figure out the composition of carotid artery wall through MRI images.  
Two models are applied in our project, one is the original 3D U-net model, the other is a modified version of 3D U-Net, called isensee model, which is more complicated and applies instance normalization in certain layers, of course, with a more desirable performance.

## Use commands below to train the model:
cd brats  
python train.py  
  
  PS：You can set "if_isensee2017_model" in "configure" to choose the model to be trained.
