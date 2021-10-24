import numpy as np
#import cv2
import glob
from PIL import Image
import os
import argparse

#from basic_edge_detection import edge_detect

from skimage import filters, img_as_ubyte
from skimage.feature import canny
from skimage.color import rgb2gray

parser = argparse.ArgumentParser(description='Program to preprocess images.')
parser.add_argument("dataset_location", help="File path to dataset of images.", default=None)
args = parser.parse_args()

# Function that takes in the original flower data set and normalizes the image sizes.
# Saves it in new_image_destination.
def create_crop_and_edge_images(image_location, dim = 256, batch_size = 32):
  #modify and save images
  count = 0
  images = []
  for f in glob.iglob(image_location + '\*.jpg'):
    with Image.open(f) as im:
      im_crop = im.copy() #make copy to crop
      w, h = im_crop.size
      min_d = min(w, h)
      im_crop = im_crop.crop((0, 0, min_d, min_d)) #square crop
      im_crop = im_crop.resize((dim, dim)) #scale to output size

      edge_im = im_crop.copy() #make copy to edge detect
      edge_im = rgb2gray(np.array(edge_im))
      edge_im = Image.fromarray(img_as_ubyte(canny(edge_im, sigma=3)))

      images.append((im_crop, edge_im))
      count += 1
      if(count % batch_size == 0 or count == len(f)-1):
        yield images
        images = []

def concat_images(crop_images, edge_images, new_image_destination, count):
  for i in range(len(crop_images)):
    #filename = edge[i]
    crop_img = (crop_images[i])
    edge_img = (edge_images[i])
    
    total_width = (edge_img.width + crop_img.width)
    max_height = max(edge_img.height, crop_img.height)

    new_im = Image.new('RGB', (total_width, max_height))
    new_im.paste(crop_img, (0, 0))

    new_im.paste(edge_img, (crop_img.width, 0))
    new_im.save(new_image_destination + '\\' + "image_"+str(count)+'.jpg', quality=95)
    count += 1
  return count

# Preprocessing steps:
# Given a file path to dataset of images:
# 1. Crop images, save in new location.
# 2. Run edge detect on images, save in new location.
# 3. Concatenate images, save in new location.

def main():
  dataset_location = args.dataset_location
  if(dataset_location == None or not os.path.isdir(dataset_location)):
    print("ERROR: Path to dataset does not exist or is not a directory")
    return None
  new_image_destination = os.path.dirname(dataset_location) + "/preprocessed_data"
  if not os.path.exists(new_image_destination):
      os.makedirs(new_image_destination)
  batch_size = 1 #files get cut off if this number is larger?
  images_gen = (c_image for c_image in create_crop_and_edge_images(dataset_location, batch_size=batch_size))
  count = 1
  for c_images in images_gen:
    crop_imgs = [pairs[0] for pairs in c_images]
    edge_imgs = [pairs[1] for pairs in c_images]
    count = concat_images(crop_imgs, edge_imgs, new_image_destination, count)
        
if __name__ == '__main__':
   main()
