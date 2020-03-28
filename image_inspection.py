import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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


def neural_network():
    batch_size = 128
    epochs = 15
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )

    validation_image_generator = ImageDataGenerator(rescale=1./255)

    # sljedeci dio koda samo radi train_dir
    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    path_to_zip = tf.keras.utils.get_file('/home/kemal/Programming/Python/Image_preprocessing_for_ANNs/catsNdogs/cats_and_dogs.zip', origin=_URL, extract=True)
    PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')
    train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                         directory=train_dir,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode='binary')
    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=validation_dir,
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='binary')

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
               input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    total_train = 2000
    total_val = 1000

    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=3.0)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=3.0)
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', linewidth=3.0)
    plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=3.0)
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


fetching_data()
plot_images([i for i in range(500, 509)])
neural_network()

