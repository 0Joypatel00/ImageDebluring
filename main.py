import sys
import keras
import cv2
import numpy
import matplotlib
import skimage
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
import cv2
import numpy as np
import math
import os

"""print('Python: {}'.format(sys.version))
print('Keras: {}'.format(keras.__version__))
print('OpenCV: {}'.format(cv2.__version__))
print('NumPy: {}'.format(numpy.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Scikit-Image: {}'.format(skimage.__version__))

"""
# define a function for peak signal-to-noise ratio (PSNR)
def psnr(target, ref):
    # assume RGB image
    target_data = target.astype(float)
    ref_data = ref.astype(float)
    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)


# define function for mean squared error (MSE)
def mse(target, ref):
    # the MSE between the two images is the sum of the squared difference between the two images
    err = np.sum((target.astype('float') - ref.astype('float')) ** 2)
    err /= float(target.shape[0] * target.shape[1])

    return err


# define function that combines all three image quality metrics
def compare_images(target, ref):
    scores = []
    scores.append(psnr(target, ref))
    scores.append(mse(target, ref))
    scores.append(ssim(target, ref, multichannel=True))

    return scores


# prepare degraded images by introducing quality distortions via resizing

def prepare_images(path, factor):
    # loop through the files in the directory
    for file in os.listdir(path):

        # open the file
        img = cv2.imread(path + '/' + file)
        # print(img)

        # find old and new image dimensions
        if img is not None:
            # print('variable is not None')
            # print(img.shape)

            h, w, _ = img.shape
            dim = (w, h)
            new_height = h / 2
            new_width = w / 2
            dimnew = (int(new_height), int(new_width))
            # resize the image - down
            img = cv2.resize(img, dimnew, interpolation=cv2.INTER_LINEAR)

            # resize the image - up
            img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

            # save the image
            print('Saving {}'.format(file))
            cv2.imwrite('E:/MajorProj/majorProj/newData{}'.format(file), img)

        else:
            print('variable is None')


prepare_images('Data/', 2)

# test the generated images using the image quality metrics

for file in os.listdir('newData'):
    # open target and reference images
    target = cv2.imread('newData'.format(file))
    ref = cv2.imread('Data'.format(file))


    # calculate score
    scores = compare_images(target, ref)

    # print all three scores with new line characters (\n)
    print('{}\nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(file, scores[0], scores[1], scores[2]))