# -*- coding: utf-8 -*-
"""
Created on Fri May 29 20:59:47 2020

@author: izabi
"""

from matplotlib import pyplot as plt

from skimage import data
from skimage.feature import blob_log
from skimage.color import rgb2gray
import math
from IPython.html.widgets import interact, fixed
from skimage.io import imread


image = imread('ID55post.png')

image_gray = rgb2gray(image)

def plot_blobs(max_sigma=40, threshold=0.004, gray=False):
    """
    Plot the image and the blobs that have been found.
    """
    blobs = blob_log(image_gray, max_sigma=max_sigma, threshold=threshold)
    #blobs[:, 2] = math.sqrt(9) * blobs[:, 2]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_title('Galaxies in the Hubble Deep Field')
    
    if gray:
        ax.imshow(image_gray, interpolation='nearest', cmap='gray_r')
        circle_color = 'red'
    else:
        ax.imshow(image, interpolation='nearest')
        circle_color = 'yellow'
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=circle_color, linewidth=2, fill=False)
        ax.add_patch(c)

# Use interact to explore the galaxy detection algorithm.
interact(plot_blobs, max_sigma=(5, 15, 5), threshold=(0.1, 0.4, 0.1))