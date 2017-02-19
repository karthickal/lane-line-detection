# load the required libraries
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import numpy as np

def load_image(file):
    '''
    Method to load the image from file
    :param file: the path
    :return: the loaded image
    '''
    return convert_bgr_rgb(cv2.imread(file))

def convert_bgr_grayscale(image):
    '''
    Method to convert the color space from BGR to Grayscale
    :param image: the input image
    :return: the grayscale image
    '''
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_rgb_grayscale(image):
    '''
    Method to convert the color space from RGB to Grayscale
    :param image: the input image
    :return: the grayscale image
    '''
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def convert_bgr_rgb(bgr):
    '''
    Method to convert the color space from BGR to RGB
    :param image: the input image
    :return: the RGB image
    '''
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def convert_bgr_hsv(bgr):
    '''
    Method to convert the color space from BGR to HSV
    :param image: the input image
    :return: the HSV image
    '''
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

def convert_rgb_hsv(rgb):
    '''
    Method to convert the color space from RGB to HSV
    :param image: the input image
    :return: the HSV image
    '''
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

def convert_bgr_hls(bgr):
    '''
    Method to convert the color space from BGR to HLS
    :param image: the input image
    :return: the HLS image
    '''
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)

def convert_rgb_hls(rgb):
    '''
    Method to convert the color space from RGB to HLS
    :param image: the input image
    :return: the HLS image
    '''
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)

def compare_images(img1, img2, title1='Initial Image', title2='Final Image', fsize=(24, 9), cmap1=None, cmap2=None):
    '''
    Utility method to compare two images
    :param img1: input image 1
    :param img2: input image 2
    :param title1: title for image 1
    :param title2: title for image 2
    :param fsize: size of the figure
    :param cmap1: color map for image 1
    :param cmap2: color map for image 2
    :return:
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fsize)
    ax1.imshow(img1, cmap=cmap1)
    ax1.set_title(title1)

    ax2.imshow(img2, cmap=cmap2)
    ax2.set_title(title2)
    plt.show()


def display_image_grid(images, per_row=4, colormap=None, fsize=(40, 20)):
    '''
    Utility method to show the images in a grid
    :param images: the input image
    :param per_row: number of images per row
    :param colormap: colormap for the images
    :param fsize: figure size
    :return:
    '''
    num_rows = math.ceil(len(images) / per_row)

    fig = plt.figure(figsize=fsize)

    grid = gridspec.GridSpec(num_rows, per_row, wspace=0.0)
    ax = [plt.subplot(grid[i]) for i in range(num_rows * per_row)]
    fig.tight_layout()

    for i, img in enumerate(images):
        ax[i].imshow(img, cmap=colormap)
        ax[i].axis('off')

    plt.show()