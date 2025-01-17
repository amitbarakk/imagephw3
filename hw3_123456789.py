import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# the copy in the first lines of the function is so that you don't ruin
# the original image. it will create a new one. 

def add_SP_noise(im, p):
    sp_noise_im = im.copy()
    indices = random.sample(range(im.size), int(p * im.size))
    white_index = random.sample(range(len(indices)), int(len(indices)/2))
    black_index = np.delete(range(len(indices)), white_index)
    black = np.delete(indices, white_index)
    white = np.delete(indices, black_index)

    # Flatten the image
    img_flat = sp_noise_im.flatten()
    img_flat[white] = 0
    img_flat[black] = 255

    # Reshape the flattened image to the original dimensions (h, w)
    sp_noise_im = img_flat.reshape(sp_noise_im.shape)
    return sp_noise_im


def clean_SP_noise_single(im, radius):
    clean_im = im.copy()
    h,w = im.shape
    for i in range(h):
        for j in range(w):
            #X, Y = np.meshgrid(np.arange(i - radius, i + radius + 1), np.arange(j - radius, j + radius + 1))
            u_l = i-radius if i-radius>=0 else 0
            u_r = i+radius if i+radius<h else h-1
            l_l = j-radius if j-radius>=0 else 0
            l_r = j+radius if j+radius<w else w-1
            arr = im[u_l:u_r+1, l_l:l_r+1]
            med = np.median(arr)
            clean_im[i, j] = med
    return clean_im


def clean_SP_noise_multiple(images):
    clean_image = np.median(images, axis=0)
    return clean_image


def add_Gaussian_Noise(im, s):
    gaussian_noise_im = im.copy()
    gaus = np.random.normal(0,s,im.shape)
    gaussian_noise_im = gaussian_noise_im+gaus
    return gaussian_noise_im


def clean_Gaussian_noise(im, radius, maskSTD):
    noise_mask = np.random.normal(0, maskSTD, (radius + 1, radius + 1))

    cleaned_im = convolve2d(im, noise_mask, mode="same")

    return cleaned_im.astype(np.uint8)


def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    bilateral_im = im.copy()
    # TODO: add implementation

    return bilateral_im.astype(np.uint8)


