import tensorflow as tf
import constants
from numpy.lib.format import BUFFER_SIZE
from generator import Generator, generator_loss
from discriminator import Discriminator, discriminator_loss
from matplotlib import pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description='Program to train or test this pix2pix GAN implementation.')
parser.add_argument("-checkpoint_path", help="Path to checkpoint of model.", default=None)
parser.add_argument("-train", help="Train model.", default=True)
parser.add_argument("-test", help="Test model.", default=True)
args = parser.parse_args()


def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)
   
  edge_image = image[:, constants.IMG_WIDTH:, :]
  orig_image = image[:, :constants.IMG_WIDTH, :]
  # Normalize images to [-1, 1]
  edge_image = (tf.cast(edge_image, tf.float32) / 127.5) - 1
  orig_image =  (tf.cast(orig_image, tf.float32) / 127.5 ) - 1
  return edge_image, orig_image

def load_images():
  dataset = tf.data.Dataset.list_files(constants.IMG_FILE_PATH+'/*.jpg')
  dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)

  train_split = round(len(dataset) * constants.TRAIN_SPLIT)

  train_dataset = dataset.take(train_split) #train/test split determined by constant
  train_dataset = train_dataset.batch(constants.BATCH_SIZE)

  test_dataset = dataset.skip(train_split) # Start after where training data ends
  test_dataset = test_dataset.batch(constants.BATCH_SIZE)
  return train_dataset, test_dataset

# Show generated image to see progress of model. 
def generate_images(model, test_input, target):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], target[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

def train(input_image, target, generator, generator_optimizer, discriminator, discriminator_optimizer):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

def test(input_image, target, generator, discriminator):
  prediction = generator(input_image, training=False)
  disc_real_output = discriminator([input_image, target], training=True)
  disc_generated_output = discriminator([input_image, prediction], training=True)

  gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, prediction, target)
  disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generate_images(generator, input_image, target)

def main():                         
  generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  generator = Generator()
  discriminator = Discriminator()

  checkpoint_dir = './training_model_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  # Save model states in checkpoint.
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                  discriminator_optimizer=discriminator_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)

  checkpoint_path = args.checkpoint_path
  if(checkpoint_path != None): # If user requested a checkpoint be loaded
    if(not os.path.isdir(checkpoint_path)):
      print("ERROR: Path to checkpoint path does not exist or is not a directory")
      return
    else: # Exists, so load
      checkpoint.restore(checkpoint_path).assert_consumed()
      
  train_dataset, test_dataset = load_images()

  if(args.train): # Arg for number of epochs?
    for i in range((constants.EPOCHS)):
      train_dataset = train_dataset.shuffle(BUFFER_SIZE) #Buffer size 
      if(i % 1000 == 0): # Every 1000 images, show an output and save checkpoint.
        example_input, example_target = next(iter(test_dataset.take(1)))
        generate_images(generator, example_input, example_target) # Show example.
        checkpoint.save(file_prefix=checkpoint_prefix) # Save!
      input_image, target = next(iter(train_dataset.take(1)))
      train(input_image, target, generator, generator_optimizer, discriminator, discriminator_optimizer)
  if(args.test): # Arg for number of tests?
    input_image, target = next(iter(train_dataset.take(1)))
    test(input_image, target, generator, discriminator)

if __name__ == '__main__':
    main()
