import os as os
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import random as random
import math as math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def create_variations():
    random.seed(500)
    path = '/home/kemal/Programming/Python/Image_preprocessing_for_ANNs/catsNdogs/cats_and_dogs_filtered/train/cats/'
    entries = os.listdir(path)
    entries = entries[755:756:1]
    images = []
    for i, entry in enumerate(entries):
        images.append(Image.open(path+entry))

    save_folder = '/home/kemal/Programming/Python/Image_preprocessing_for_ANNs/PROCESIRANE_SLIKE/'
    counter = 0
    for i, img in enumerate(images):
        img.save(save_folder + 'Img_' + str(i) + '_var_' + str(counter) + '.png')
        counter += 1


        # emboss
        x = img.filter(ImageFilter.CONTOUR)
        x.save(save_folder + 'Img_' + str(i) + '_var_' + str(counter) + '.png')
        counter += 1


        # rotate clockwise
        # a = random.randint(0, 90)
        # x = img.rotate(-a)
        # x.save(save_folder + 'Img_' + str(i) + '_var_' + str(counter) + '.png')
        # counter += 1

        # transpose top-bot
        x = img.transpose(Image.FLIP_TOP_BOTTOM)
        x.save(save_folder + 'Img_' + str(i) + '_var_' + str(counter) + '.png')
        counter += 1

        # adjust brightness
        a = random.random()
        bright = ImageEnhance.Brightness(img)
        bright.enhance(a).save(save_folder + 'Img_' + str(i) + '_var_' + str(counter) + '.png')
        counter += 1

        # adjust sharpness
        a = 5 * random.random()
        x = img.transpose(Image.FLIP_TOP_BOTTOM)
        bright2 = ImageEnhance.Brightness(x)
        x = bright2.enhance(0.5)
        x = x.rotate(32)
        sharp = ImageEnhance.Color(x)
        sharp.enhance(a).save(save_folder + 'Img_' + str(i) + '_var_' + str(counter) + '.png')
        counter += 1

        # blur
        x = img.filter(ImageFilter.BLUR)
        x.save(save_folder + 'Img_' + str(i) + '_var_' + str(counter) + '.png')
        counter += 1

        # contour
        x = img.filter(ImageFilter.EMBOSS)
        x.save(save_folder + 'Img_' + str(i) + '_var_' + str(counter) + '.png')
        counter += 1

        # smooth_more
        x = img.filter(ImageFilter.SMOOTH_MORE)
        x.save(save_folder + 'Img_' + str(i) + '_var_' + str(counter) + '.png')
        counter += 1


        # rotate counter-clockwise
        a = random.randint(0, 90)
        x = img.rotate(a)
        x.save(save_folder + 'Img_' + str(i) + '_var_' + str(counter) + '.png')
        counter += 1

        # img.show()
        #
        # # rotate counter-clockwise
        # a = random.randint(0, 90)
        # x = img.rotate(a)
        # x.show()
        #
        # # rotate clockwise
        # a = random.randint(0, 90)
        # x = img.rotate(-a)
        # x.show()
        #
        # # transpose top-bot
        # x = img.transpose(Image.FLIP_TOP_BOTTOM)
        # x.show()
        #
        # # transpose left-right
        # x = img.transpose(Image.FLIP_LEFT_RIGHT)
        # x.show()
        #
        # # adjust sharpness
        # a = 2 * random.random()
        # sharp = ImageEnhance.Sharpness(img)
        # sharp.enhance(a).show()
        #
        # a = 2 * random.random()
        # sharp = ImageEnhance.Sharpness(img)
        # sharp.enhance(a).show()
        #
        # # adjust brightness
        # a = random.random()
        # bright = ImageEnhance.Brightness(img)
        # bright.enhance(a).show()
        #
        # a = random.random()
        # bright = ImageEnhance.Brightness(img)
        # bright.enhance(a).show()


def plot_images():
    plt.figure(figsize=(15, 15))
    for i in range(9):
        img = mpimg.imread('/home/kemal/Programming/Python/Image_preprocessing_for_ANNs/PROCESIRANE_SLIKE/Img_0_var_'+str(i)+'.png')
        plt.subplot(3, 3, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img)
        plt.xlabel('Variation: ' + str(i))
    plt.show()




create_variations()
plot_images()