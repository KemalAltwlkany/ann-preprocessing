import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import tensorflow as tf


# noinspection DuplicatedCode
def fetching_data():
    # loading the dataset, we use the Dogs vs Cats dataset from Kaggle
    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    path_to_zip = tf.keras.utils.get_file('/home/kemal/Programming/Python/Image_preprocessing_for_ANNs/catsNdogs/cats_and_dogs.zip', origin=_URL, extract=True)
    print(path_to_zip)
    print(type(path_to_zip))
    PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')
    train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

    num_cats_tr = len(os.listdir(train_cats_dir))
    num_dogs_tr = len(os.listdir(train_dogs_dir))

    num_cats_val = len(os.listdir(validation_cats_dir))
    num_dogs_val = len(os.listdir(validation_dogs_dir))

    total_train = num_cats_tr + num_dogs_tr
    total_val = num_cats_val + num_dogs_val

    print('total training cat images:', num_cats_tr)
    print('total training dog images:', num_dogs_tr)

    print('total validation cat images:', num_cats_val)
    print('total validation dog images:', num_dogs_val)
    print("--")
    print("Total training images:", total_train)
    print("Total validation images:", total_val)


def plot_images(indices_to_plot):
    plt.figure(figsize=(15, 15))
    for ind, i in enumerate(indices_to_plot):
        img = mpimg.imread('/home/kemal/Programming/Python/Image_preprocessing_for_ANNs/catsNdogs/cats_and_dogs_filtered/train/cats/cat.'+str(i)+'.jpg')
        plt.subplot(3, 3, ind+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img)
        plt.xlabel('cat_' + str(i))
    plt.show()


fetching_data()
plot_images([i for i in range(500, 509)])