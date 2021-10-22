# Constants for run_gan
IMG_FILE_PATH ='../data/preprocessed_data'
BATCH_SIZE = 1 # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment

#BUFFER_SIZE = 8189
IMG_WIDTH = 256 # Each image is 256x256 in size
IMG_HEIGHT = 256
TRAIN_SPLIT = .7
EPOCHS = 1
# For generator loss:
LAMBDA = 100
# For generator:
OUTPUT_CHANNELS = 3
