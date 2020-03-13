import numpy as np
from PIL import Image




def example1():
    """
    Example of reading the batches.meta file. This file contains a python dictionary, which contains
    3 useful pieces of information. All keys in the dictionary are byte-strings and can be accessed as:
        dict[b'name']
    The keys are:
        b'num_cases_per_batch', containing obviously the number of data instances used per batch
        b'label_names', containing a python list whose elements are byte-strings. They are meaningful names
        for the classes/labes of items being used.
    example1() opens the file "batches.meta" and prints the keys and values of the dictionary.
    """
    dictx = unpickle('/home/kemal/Programming/Python/Image_preprocessing_for_ANNs/cifar-10-batches-py/batches.meta')
    for k, v in dictx.items():
        print("key = ", k, "    value = ", v)
    print(dictx[b'label_names'][0])
    return dictx


def example2():
    """
    Shows example of opening a batch-file and iterating through the dictionary containing it.
    :return:
    """
    dictx = unpickle('/home/kemal/Programming/Python/Image_preprocessing_for_ANNs/cifar-10-batches-py/data_batch_1')
    for k, v in dictx.items():
        print("key = ", k, "    value = ", v)
    return dictx


def show_images(data_dict, meta_dict, indices):
    """

    :param indices:
    :return:
    """
    raw_images = data_dict[b'data']
    labels = data_dict[b'labels']
    label_names = meta_dict[b'label_names']

    for ind, i in enumerate(indices):
        single_image = np.transpose(np.reshape(raw_images[i], (3, 32, 32)), (1, 2, 0))
        img =  Image.fromarray(single_image)
        img.save('/home/kemal/Programming/Python/Image_preprocessing_for_ANNs/images/img_'+str(ind)+str(label_names[labels[i]], 'utf-8')+'.png')




if __name__ == '__main__':
    dict1 = example1()
    dict2 = example2()
    show_images(dict2, dict1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])