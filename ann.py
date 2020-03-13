import matplotlib.pyplot as plt
import pickle
from PIL import Image
import numpy as np


def unpickle(fileName):
    """
    Function opens a binary file and returns the dictionary encoded within it.
    :param fileName: name of file in form of string, preceded by full path. Example:
        '/home/kemal/Programming/Python/Image_preprocessing_for_ANNs/cifar-10-batches-py/batches.meta'
    :return: python dictionary object.
    """
    with open(fileName, 'rb') as fo:
        dictx = pickle.load(fo, encoding='bytes')
    return dictx


def load_batch(fileName):
    dictx = unpickle(fileName)
    data = dictx[b'data']
    labels = dictx[b'labels']
    return data, labels


def load_labels(fileName):
    dictx = unpickle(fileName)
    label_names = dictx[b'label_names']
    for i in range(len(label_names)):
        label_names[i] = str(label_names[i], 'utf-8')
    return label_names


def data_verification():
    data, labels = load_batch('/home/kemal/Programming/Python/Image_preprocessing_for_ANNs/cifar-10-batches-py'
                              '/data_batch_1')
    label_names = load_labels('/home/kemal/Programming/Python/Image_preprocessing_for_ANNs/cifar-10-batches-py'
                              '/batches.meta')

    print(label_names)
    print(data[0])
    print(labels[0])

    ###

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img = np.transpose(np.reshape(data[i], (3, 32, 32)), (1, 2, 0))
        plt.imshow(img)
        plt.xlabel(label_names[labels[i]])
    plt.show()


def rgb_to_grayscale(img_array):
    """
    :param img_array: a 1D np array of shape (3072, ). The first 1024 elements correspond to the red channel values,
    the next 1024 elements to the green entries and the last 1024 elements to the blue component (RGB image).
    :return:
    """
    # img = np.transpose(np.reshape(img_array, (3, 32, 32)), (1, 2, 0))
    r, g, b = img_array[0:1024], img_array[1024:2048], img_array[2048:]
    y = 0.2126*r + 0.7152*g + 0.0722*b
    img = np.reshape(y, (1, 32, 32))
    img = np.mean(img, axis=0)
    print(img.shape)
    return img


def demo():
    data, labels = load_batch('/home/kemal/Programming/Python/Image_preprocessing_for_ANNs/cifar-10-batches-py'
                              '/data_batch_1')
    label_names = load_labels('/home/kemal/Programming/Python/Image_preprocessing_for_ANNs/cifar-10-batches-py'
                              '/batches.meta')

    # img = data[0]
    for i in range(10):
        img = rgb_to_grayscale(data[i])
        img = Image.fromarray(img)
        img.show()


def demo2():
    x = np.array([ [[1, 1, 1], [1, 1, 1]], [ [2, 2, 2], [2, 2, 2]], [[3, 3, 3], [3, 3, 3]] ])
    print(x)
    print(x.shape)
    x = rgb_to_grayscale(x)
    print(x)
    print(x.shape)


# data_verification()
demo()
# demo2()
