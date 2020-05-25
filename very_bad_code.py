# -*- coding: utf-8 -*-
"""
Created on Wed May 20 18:31:05 2020

@author: izabi
"""
import argparse
import imutils
import numpy as np
from imageio import imread
import matplotlib
from matplotlib import pyplot as plt
from skimage.io import imread
import morphsnakes as ms
import cv2
import scipy
from scipy import ndimage

def rgb2gray(img):
    return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def visual_callback_2d(background, fig=None):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 2D images.
    
    
    """
    
    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    ax_u = ax1.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(levelset):
        
        if ax1.collections:
            del ax1.collections[0]
        #ax1.contour(levelset, [0.5], colors='b')
        ax_u.set_data(levelset)
        fig.canvas.draw()
        plt.pause(0.001)
        

    return callback

def sharpShape(image, calback):
    """
     `morphological_chan_vese` with using the default
    initialization of the level-set.
    """
    
   
    
    # Callback for visual plotting
    callback = visual_callback_2d(image)

    # Morphological Chan-Vese 
    ima=ms.morphological_chan_vese(image, 35,
                               smoothing=3, lambda1=1, lambda2=1,
                               iter_callback=callback)
    return ima
def crop_image(img):
    '''
    Crops image to remove text in top left/right corner.
    '''
    crop_img = img[50:img.shape[0], 0:img.shape[1]]
    return crop_img

def normalize(img):
    '''
    Normalize image so all pixel values are between 0 and 1.
    '''
    min_pixel = np.amin(img)
    max_pixel = np.amax(img)
    norm_img = (img - min_pixel) * (1 / (max_pixel - min_pixel))
    return norm_img

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask
    
im= imread("ID07pre.png",0)
im_crop=crop_image(im)
img=normalize(im_crop)
def gray_im(img):
    if (len(img.shape) == 3):
        image=rgb2gray(img)
    else:
        image=img
    return image

def center(img):
    bin2gray = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    _,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    point = cv2.HoughCircles(th,cv2.HOUGH_GRADIENT,1,800,
                            param1=100,param2=15,minRadius=50,maxRadius=150)
    point = np.uint16(np.around(point))
    for i in point[0,:]:
    #the center 
        cv2.circle(bin2gray,(i[0],i[1]),2,(255,0,0),5)
    plt.imshow(bin2gray)
    return point



image = gray_im(img)
call=visual_callback_2d(image)
binar = sharpShape(image, call)
plt.imsave('im07.png',binar)

im_bin= cv2.imread("im07.png",0)
im_bin = cv2.normalize(im_bin,  None, 0, 255, cv2.NORM_MINMAX)
im_bin = cv2.medianBlur(im_bin,13)
h, w = im_bin.shape[:2]
mask = create_circular_mask(h, w)
masked_img = im_bin.copy()
masked_img[~mask] = 0
center_point=center(masked_img)



#p_add = cv2.goodFeaturesToTrack(im,12,0.001,100)
#binar_float = np.uint8(binar)
#slicecanny = cv2.Canny(binar_float,0,1)
#plt.imshow(binar)
#distance_map = ndimage.distance_transform_edt(binar)
#plt.imshow(distance_map)
#plt.imsave('image.png', binar)

#plt.imshow(binar)
#plt.show(p_add)
