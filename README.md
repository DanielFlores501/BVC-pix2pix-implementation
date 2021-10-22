# BVC-pix2pix-implementation

# Introduction
This project uses a conditional GAN to translate an image of edges into a more realistic photo, based on the work done in this [2016 Paper titled: Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004). The generator architecture draws from the U-Net architecture used in the paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) with the addition of skips in the architecture to augment the outputs as described in the paper.

# Running the application
To train and/or test the model, run **run_gan.py** with the additional flags: '--train True' and '--test True', or to load a saved trained model, '--checkpoint_path <path>'. 

# Data

# Credits

# Training

