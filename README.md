# BVC-pix2pix-implementation

# Introduction
This project uses a conditional GAN to translate an image of edges into a more realistic photo, based on the work done in this [2016 Paper titled: Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004). By utilizing a Conditional GAN, we are able to to extract an edge map from an image, train the GAN on that edge map, and condition the generator to create images that appear similar to the original image. 

# Methodology
The generator architecture draws from the U-Net architecture used in the paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), with the addition of skips connections in the architecture to augment the outputs by connecting low level information from the inputs to the outputs, such as prominent edges as described in the paper. The discriminator architecture, termed PatchGAN by the pix2pix paper, determines if every N x N patch in the image is real or fake. By classsifying patches of the image rather than the full image at once, this method also assumes that pixels further than one patch apart are independent, treating the image as a Markov random field. 
The helper methods that streamline the architeture creation was inspired by [this guide](https://www.tensorflow.org/tutorials/generative/pix2pix).

# Data
The architecture in this model was designed to deal with a variety of pixel to pixel problems, but for this project I decided to focus on reconstructing objects from their edge maps. Specifically, I wanted to challenge the model to learn the diverse array of flower shapes, and reconstruct a natural looking flower just from its edges.

The dataset I used to train and test the model was the [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). To preprocess the data, I created a generalized script in **preprocess.py** to prepare any dataset of images. First the images are read in and cropped to square dimensions, then resized to a uniform size to preserve the image. These images are then run through the canny edge detection algorithm from the feature module of skimage with a sigma value of 3 to keep the strongest edges. The original cropped image and the edge map are concatenated and saved in a new folder in the same directory as the data. The image pairing was inspired by [this guide](https://www.tensorflow.org/tutorials/generative/pix2pix) and adapted for this project. 

# Running the application
To preprocess a dataset, run **preprocess.py** with the required flag: '--dataset_location <path\to\data>'. This will create a new folder with the same amount of images

To train and/or test the model, run **run_gan.py** with the additional flags: '--train True' and '--test True', or to load a saved trained model, '--checkpoint_path <path>'. 

# Training and Results
Work in progress!
