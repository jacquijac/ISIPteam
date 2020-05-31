from scipy import signal
from skimage import io, feature, color, filters
import matplotlib.pyplot as plt
import cv2
import numpy as np

from skimage import draw, measure

from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_fill_holes
from scipy.ndimage.morphology import morphological_gradient, distance_transform_edt
from skimage import morphology as morph
import scipy.fftpack as fp
import numpy.fft
import imutils
from imageio import imread
from matplotlib import pyplot as plt
from skimage.io import imread
import morphsnakes as ms #pip install morphsnakes
import scipy
from scipy import ndimage, spatial
import openpyxl as pyx

#meagans functions
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import feature, color, filters, draw, measure
from scipy import ndimage, spatial
from PIL import Image
import cv2
import argparse
import imutils
from imutils import contours
import math
from operator import itemgetter

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

#crop and normalize pre and post images
pre_crop = crop_image(pre)
post_crop = crop_image(post)
post_norm = normalize(post_crop)

def find_bright_points(image):
    '''
    Find brightest points in image. Function uses thresholding, based on the 
    mean pixel of an image, to extract the brightest pixels. 
    INPUT: 
        - image: post-operative image 
    OUTPUT: 
        - thresh_img: mask image showing pixels between determined threshold and 1
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_pixel = np.mean(gray)
    if mean_pixel < 0.1:
        thresh = 0.155
    elif mean_pixel > 0.1 and mean_pixel < 0.2:
        thresh = 0.35
    elif mean_pixel > 0.2 and mean_pixel < 0.285:
        thresh = 0.5
    elif mean_pixel > 0.285 and mean_pixel < 0.5:
        thresh = 0.8
    elif mean_pixel > 0.5:
        thresh = 0.99
    thresh_img = cv2.threshold(gray, thresh, 1, cv2.THRESH_BINARY)[1]
    # perform a series of erosions and dilations to remove
    # any small blobs of noise from the thresholded image
    thresh_img = cv2.erode(thresh_img, None, iterations=6)
    thresh_img = cv2.dilate(thresh_img, None, iterations=4)
    
    return thresh_img 

def select_spiral(thresh_img, center):
    '''
    Remove detected components that are more than 400 pixels left or right from the spiral center. 
    INPUTS:
        - thresh_img: thresholded image showing contours (bright spots) - output from find_bright_points function
        - center: coordinates of spiral center
    OUTPUTS:
        - thresh_img_spiral: thresholded image containing only components within 400 of spiral center
    '''
    # copy thresh_img so original image won't be changed
    thresh_img_spiral = thresh_img.copy()
    # if spiral center is less than 400 pixels from either boundary, don't need to remove from that side
    # otherwise, set all pixels from 0 to center-400 and center+400 to end equal to zero
    if center[0] > 400 and center[0] < thresh_img.shape[1]-400: 
        thresh_img_spiral[:, :int(center[0])-400] = 0
        thresh_img_spiral[:, int(center[0])+400:] = 0
    elif center[0] < 400 and center[0] < thresh_img.shape[1]-400:
        thresh_img_spiral[:, int(center[0])+400:] = 0
    elif center[0] > 400 and center[0] > thresh_img.shape[1]-400:
        thresh_img_spiral[:, :int(center[0])-400] = 0
    return thresh_img_spiral

def find_components(mask):
    '''
    Function finds all connected components (white parts of image), then determines which section of the image
    contains the most components. The components that are not in this section are blacked out. 
    INPUT:
        - mask: mask image containing brightest points
    OUTPUT:
        - comp_img: image containing only components around spiral center
    '''
    #create empty list to store number of components in each section
    num_comps = list()
    #create copy of mask so original doesn't get changed 
    mask_img = mask.copy()
    mask_img = np.uint8(mask_img)
    #loop over columns of mask
    for i in range(0, mask_img.shape[1]):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_img[:,i:i+400], connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #remove the background component
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        num_comps.append(nb_components)

    #find index of section with most components
    most_comps = num_comps.index(max(num_comps))
    #index corresponds to index:index+400 on image
    #black out everything except that section 
    mask_img[:, 0:most_comps] = 0
    mask_img[:, (most_comps)+400:] = 0

    #now get stats for only this section  
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_img, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    #we only want to keep the smallest components since they tend to form a circle around the spiral center
    #also want to get rid of really small components, since they are likely just noise 
    min_size = np.mean(sizes)*0.2
    max_size = np.mean(sizes)*1.33

    #initialize output image with zeros 
    comp_img = np.zeros((output.shape))
    
    #for every component in the image, you keep it only if it's below max_size
    for i in range(0, nb_components):
        if sizes[i] <= max_size and sizes[i] >= min_size:
            comp_img[output == i + 1] = 255
            
    return comp_img


def find_center(comp_img):
    '''
    Get x and y coordinates of all remaining components and find means of coordinates 
    which will represent the center of the cochlear spiral.
    INPUTS:
        - comp_img: image of connected components 
    OUTPUTS:
        - xmean: x coordinate of center
        - ymean: y coordinate of center
    '''
    comp_img = np.uint8(comp_img)
    y,x = np.nonzero(comp_img)

    #find mean of x and y, which represents the center of the electrodes circling the spiral center
    xmean = x.mean()
    ymean = y.mean()
    
    return xmean, ymean



def find_electrodes(image, thresh_img, center):
    '''
    Perform a connected component analysis on the thresholded
    image, then initialize a mask to store only the relevant components.
    INPUTS:
        - image: post image on which electrodes are to be labeled
        - thresh_img: thresholded image, showing brightest points
        - center: coordinates of the spiral center
    OUTPUT:
        - out_image: copy of input image, with electrodes outlined
        - cnt_coords: list of coordinates of contours
    '''

    thresh_img = np.uint8(thresh_img)
    out_image = image.copy()
    labels = measure.label(thresh_img, neighbors=8, background=0)
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the number of pixels 
        labelMask = np.zeros(thresh_img.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        # if the number of pixels in the component is not too large, add it to mask
        if numPixels < 4500:
            mask = cv2.add(mask, labelMask)
        
    # find the contours in the mask, then sort them from bottom to top
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnt_coords = []
    # loop over the contours
    for c in cnts:
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cnt_coords.append((cX, cY))
        cv2.circle(out_image, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)
        
    return out_image, cnt_coords

def dist_between_points(x1, y1, x2, y2):
    '''
    Calculate distance between two points.
    INPUTS:
        - x1, y1: x and y coordinates of point 1
        - x2, y2: x and y coordinates of point 2
    OUTPUTS: 
        - dist: distance between points
    '''
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist

def spiral_direction(thresh_img, center):
    '''
    Determine the direction of the spiral. If tail goes right, direction is right 
    and if tail goes left, direction is left. 
    INPUTS:
        - thresh_img: mask image containing all connected components 
        - center: coordinates of spiral center
    OUTPUT:
        - spiral_dir: string indicating "right" or "left" direction 
    '''
    # find coordinates of centroids of connected components and sort them from highest to lowest y value
    thresh_img = np.uint8(thresh_img)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, connectivity=8)
    cnt_coords = centroids[1:]
    cnt_coords = sorted(centroids, key=itemgetter(1), reverse = True)
    # get coordinates of 4 lowest electrodes 
    # Note: coords are sorted from bottom to top so bottom 4 are first 4 in list
    lowest = cnt_coords[0:4]
    # Find rightmost and leftmost coordinate of the 4 lowest
    bottom_right = cnt_coords[np.argmax([lowest[i][0] for i in range(len(lowest))])]
    bottom_right = (bottom_right[0], bottom_right[1])
    bottom_left = cnt_coords[np.argmin([lowest[i][0] for i in range(len(lowest))])]
    bottom_left = (bottom_left[0], bottom_left[1])
    # calculate distance from bottom right and bottom left coordinate to the center
    center_to_right = dist_between_points(bottom_right[0], bottom_right[1], center[0], center[1]) 
    center_to_left = dist_between_points(bottom_left[0], bottom_left[1], center[0], center[1]) 
    center_to_lowest = dist_between_points(lowest[0][0], lowest[0][1], center[0], center[1]) 
    # the one furthest from the center will determine the spiral direction
    if center_to_right > center_to_left:
        spiral_dir = 'right'
    else: 
        spiral_dir = 'left'
    
    return spiral_dir

def get_next_elec(cnt_coords, elec_coords, direction, median_dist):
    '''
    Determines coordinate of next electrode in spiral. 
    INPUTS:
        - cnt_coords: list of coordinates of contours
        - elec_coords: list of coordinates of electrodes that have already been identified 
        - direction: direction to look for next electrode (i.e. "left", "right", "up", or "down")
        - median_dist: approximate distance between electrodes (electrodes should be equally spaced)
    OUTPUTS: 
        - cnt_coords: updated list of contour coordinates with coordinate of next electrode deleted
        - elec_coords: updated list of electrode coordinates with coordinate of next electrode added 
    '''
    # determine whether to add/subtract in x or y direction 
    if direction == "left":
        x = -median_dist
        y = 0
    elif direction == "up":
        x = 0
        y = -median_dist
    elif direction == "right":
        x = median_dist
        y = 0
    elif direction == "down":
        x = 0
        y = median_dist
    # look for next electrode in specified direction from last identified electrode
    next_elec = (elec_coords[-1][0]+x, elec_coords[-1][1]+y)
    # use a search tree to avoid doing many distance calculations 
    tree = spatial.KDTree(cnt_coords)
    next_elec_index = tree.query(next_elec)
    next_elec_coord = cnt_coords[next_elec_index[1]]
    # add new electrode coordinate to elec_coords and remove it from cnt_coords so it cannot be identified again 
    elec_coords.append(next_elec_coord)
    cnt_coords.remove(next_elec_coord)

    return cnt_coords, elec_coords


def enumerate_electrodes(cnt_coords, center, thresh_img):
    '''
    Enumerate electrodes by looking for next electrode in a spiral-like shape. 
    (i.e. if spiral direction is right, first look
    left, then up, then right, then down, then left) 
    INPUTS: 
        - cnt_coords: list of coordiantes of contours
        - center: coordinates of spiral center
    OUTPUT: 
        - elec_coords: list of electrode coordinates, ordered from spiral tail to spiral center
    '''
    # determine spiral direction and bottom right and left electrodes
    spiral_dir = spiral_direction(thresh_img, center)
    lowest = cnt_coords[0:4]
    bottom_right = cnt_coords[np.argmax([lowest[i][0] for i in range(len(lowest))])]
    bottom_left = cnt_coords[np.argmin([lowest[i][0] for i in range(len(lowest))])]
    # initialize list to store ordered electrode coordinates
    elec_coords = []
    # initialize array to store distances between electrodes 
    distances = np.empty([len(cnt_coords), len(cnt_coords)])
    # calculate distances between all electrodes 
    for i in range(len(cnt_coords)):
        for j in range(len(cnt_coords)):
            distances[i, j] = dist_between_points(cnt_coords[i][0], cnt_coords[i][1], cnt_coords[j][0], cnt_coords[j][1])
    # calculate median distance - this will be used to approximate distance between consecutive electrodes
    # multiply by 0.6, since median tends to be a slight overestimate
    median_dist = np.median(distances)*0.6
    # the directions in which to search for electrodes are diferent for left and right spirals
    if spiral_dir == "right":
        # set last electrode as bottom_right, add it to elec_coords and remove it from cnt_coords
        last_elec = bottom_right
        elec_coords.append(last_elec)
        cnt_coords.remove(last_elec)
        # we can use get_next_elec() function as long as there are more than 2 coordinates left to search from
        while len(cnt_coords)>2:
            # in each iteration, find the leftmost, rightmost, highest and lowest electrodes
            leftmost = np.amin([cnt_coords[j][0] for j in range(len(cnt_coords))])
            highest = np.amin([cnt_coords[j][1] for j in range(len(cnt_coords))])
            rightmost = np.amax([cnt_coords[j][0] for j in range(len(cnt_coords))])
            lowest = np.amax([cnt_coords[j][1] for j in range(len(cnt_coords))])
            # search for next electrode in left direction until we get to the leftmost electrode
            while elec_coords[-1][0] != leftmost and len(cnt_coords)>2:
                cnt_coords, elec_coords = get_next_elec(cnt_coords, elec_coords, "left", median_dist)
            # now search up until we get to the highest electrode
            while elec_coords[-1][1] != highest and len(cnt_coords)>2:
                cnt_coords, elec_coords = get_next_elec(cnt_coords, elec_coords, "up", median_dist)
            # search right until rightmost
            while elec_coords[-1][0] != rightmost and len(cnt_coords)>2:
                cnt_coords, elec_coords = get_next_elec(cnt_coords, elec_coords, "right", median_dist)
            # search down until lowest
            while elec_coords[-1][1] != lowest and len(cnt_coords)>2:
                cnt_coords, elec_coords = get_next_elec(cnt_coords, elec_coords, "down", median_dist)
                break
            # search left until leftmost
            while elec_coords[-1][0] != leftmost and len(cnt_coords)>2:
                cnt_coords, elec_coords = get_next_elec(cnt_coords, elec_coords, "left", median_dist)
                break
        # once there are only 2 coordinate left in cnt_coords, we can assume the one closest to the 
        # previous electrode is the next electrode
        if len(cnt_coords) == 2:
            # calculate distance between previous electrode and 2 possible next electrodes
            distance1 = dist_between_points(cnt_coords[0][0], cnt_coords[0][1], elec_coords[-1][0], elec_coords[-1][1])
            distance2 = dist_between_points(cnt_coords[1][0], cnt_coords[1][1], elec_coords[-1][0], elec_coords[-1][1])
            # next electrode will be one with smallest distance
            next_elec_index = np.argmin([distance1, distance2])
            next_elec_coord = cnt_coords[next_elec_index]
            # append it to elec_coords and delete from cnt_coords
            elec_coords.append(next_elec_coord)
            cnt_coords.remove(next_elec_coord)
            # now we can add last one
            elec_coords.append(cnt_coords[0])
            cnt_coords.remove(cnt_coords[0])
    # do the same for left spirals, now looking right, then up, then left, then down, then right
    elif spiral_dir == "left":
        last_elec = bottom_left
        elec_coords.append(last_elec)
        cnt_coords.remove(last_elec)
        while len(cnt_coords)>2:
            leftmost = np.amin([cnt_coords[j][0] for j in range(len(cnt_coords))])
            highest = np.amin([cnt_coords[j][1] for j in range(len(cnt_coords))])
            rightmost = np.amax([cnt_coords[j][0] for j in range(len(cnt_coords))])
            lowest = np.amax([cnt_coords[j][1] for j in range(len(cnt_coords))])
            while elec_coords[-1][0] != rightmost and len(cnt_coords)>2:
                cnt_coords, elec_coords = get_next_elec(cnt_coords, elec_coords, "right", median_dist)
                #break 
            while elec_coords[-1][1] != highest and len(cnt_coords)>2:
                cnt_coords, elec_coords = get_next_elec(cnt_coords, elec_coords, "up", median_dist)
                #break
            while elec_coords[-1][0] != leftmost and len(cnt_coords)>2:
                cnt_coords, elec_coords = get_next_elec(cnt_coords, elec_coords, "left", median_dist)
                #break
            while elec_coords[-1][1] != lowest and len(cnt_coords)>2:
                cnt_coords, elec_coords = get_next_elec(cnt_coords, elec_coords, "down", median_dist)
                break
            while elec_coords[-1][0] != rightmost and len(cnt_coords)>2:
                cnt_coords, elec_coords = get_next_elec(cnt_coords, elec_coords, "right", median_dist)
                break
        if len(cnt_coords) == 2:
            distance1 = dist_between_points(cnt_coords[0][0], cnt_coords[0][1], elec_coords[-1][0], elec_coords[-1][1])
            distance2 = dist_between_points(cnt_coords[1][0], cnt_coords[1][1], elec_coords[-1][0], elec_coords[-1][1])
            next_elec_index = np.argmin([distance1, distance2])
            next_elec_coord = cnt_coords[next_elec_index]
            elec_coords.append(next_elec_coord)
            cnt_coords.remove(next_elec_coord)
            elec_coords.append(cnt_coords[0])
            cnt_coords.remove(cnt_coords[0])
                         
    return elec_coords


#old functions






def create_circular_mask(h, w, center=None, radius=None): 

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask
#it occurred to me so that maybe it would work better with a mask in the shape of a circle 
#- then the electrodes do not cut off, and in some images this glowing tail is cut off
#h, w = img.shape[:2]
#mask = create_circular_mask(h, w)
#masked_img = image.copy()
#masked_img[~mask] = 0

def find_components(mask):
    '''
    Function finds all connected components (white parts of image), then determines which section of the image
    contains the most components. The components that are not in this section are blacked out. 
    '''
    #create empty list to store number of components in each section
    num_comps = list()
    #create copy of mask so original doesn't get changed 
    mask_img = mask.copy()
    mask_img = np.uint8(mask_img)
    #loop over columns of mask
    for i in range(0, mask_img.shape[1]):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_img[:,i:i+400], connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #remove the background component
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        num_comps.append(nb_components)

    #find index of section with most components
    most_comps = num_comps.index(max(num_comps))
    #index corresponds to index:index+400 on image
    #black out everything except that section 
    mask_img[:, 0:most_comps] = 0
    mask_img[:, (most_comps)+400:] = 0


    #now get stats for only this section  
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_img, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    #we only want to keep the smallest components since they tend to form a circle around the spiral center
    #take maximum size to be (4/3)*mean(size)
    min_size = np.mean(sizes)*0.2
    max_size = np.mean(sizes)*1.33

    #initialize output image with zeros 
    comp_img = np.zeros((output.shape))
    
    #for every component in the image, you keep it only if it's below max_size
    for i in range(0, nb_components):
        if sizes[i] <= max_size and sizes[i] >= min_size:
            comp_img[output == i + 1] = 255
            
    return comp_img


def find_center(comp_img):
    '''
    Get x and y coordinates of all remaining components and find means of coordinates.
    '''
    y,x = np.nonzero(comp_img)

    #find mean of x and y, which represents the center of the electrodes circling the spiral center
    xmean = x.mean()
    ymean = y.mean()
    
    return xmean, ymean


def circle_convolution(image, radius, kernelsize):
    """
    function to convolve img with a kernel containing a circle filter
    kernelsize= size of kernel
    radius= radius of circle
    """
    
    arr = np.zeros((kernelsize, kernelsize))
    arr[:,:] = -100
    rr, cc = draw.circle(kernelsize/2, kernelsize/2, radius=radius, shape=arr.shape)
    arr[rr, cc] = 1000
    
    conv = signal.convolve2d(image, arr, mode='same')
    return conv


def thresholding(image, thres):
    """
    function for thresholding
    """
    binary_img=np.where(image>thres, 200, 0)
    return binary_img

def skeleton(image):
    im = color.rgb2gray(image)
    thres = filters.threshold_otsu(im) 
    im - (im > thres).astype(np.uint8)
    binary = im>thres
    skeleton=morph.skeletonize(binary)
    return skeleton


def fit_line_slope(points):
    """
    Calculates slope for a line fitted trough 4 points
    
    """

    x0=points[0][0]
    y0=points[0][1]
    x1=points[1][0]
    y1=points[1][1]
    
    #calculate x and y difference. To avoid division by zero add epsilon in x difference is zero
    y_diff=(y1-y0)
    
    x_diff=(x1-x0)
    if x_diff==0:
        x_diff=x_diff+sys.float_info.epsilon
    
    #calculate slope, if slope is zero add epsilon
    m = (y_diff)/(x_diff)
    return m

def find_insertion_angle(center, last_electrode, other_electrode):
    """
    to calculate insertion angle when center and two electrodes are given
    """
    m1 = fit_line_slope((center, last_electrode))
    m2 = fit_line_slope((center, other_electrode))
    
    tan_angle= (m1-m2)/(1+(m1*m2))
    angle= np.degrees(np.arctan(tan_angle))
    return angle

    
 def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def hpass(image_gray, l):
    '''Application of the High Pass Filter on the grayscale
image, with selected frequency cut-off values.'''
    im = np.array(image_gray)
    freq = fp.fft2(im)
    (w, h) = freq.shape
    half_w, half_h = int(w/2), int(h/2)
    freq1 = np.copy(freq)
    freq2 = fp.fftshift(freq1)
    freq2[half_w-l:half_w+l+1,half_h-l:half_h+l+1] = 0 # select all but the first lxl (low) frequencies
    im1 = np.clip(fp.ifft2(fp.ifftshift(freq2)).real,0,255) # clip pixel values after IFFT
    
    return im1

def lowpass(image_gray, u):
    '''Application of the Low Pass Filter on the grayscale
image, with selected frequency cut-off values.'''
    im = np.array(image_gray)
    freq = fp.fft2(im)
    (w, h) = freq.shape
    half_w, half_h = int(w/2), int(h/2)
    freq1 = np.copy(freq)
    freq2 = fp.fftshift(freq1)
    freq2_low = np.copy(freq2)
    freq2_low[half_w-u:half_w+u+1,half_h-u:half_h+u+1] = 0
    freq2 -= freq2_low # select only the first 20x20 (low) frequencies
    im1 = fp.ifft2(fp.ifftshift(freq2)).real
    return im1

def find_electrodes1(template, image):

    _, w, h = template.shape[::-1]
    electrodes_loc=[]
    template=color.rgb2grey(template)
    match_img=np.copy(image)
    match_img=color.rgb2grey(match_img)

    for i in range(0,12):
        match= cv2.matchTemplate(match_img, template, cv2.TM_SQDIFF_NORMED )
        min_val, max_val, min_loc, max_loc=cv2.minMaxLoc(match)
        x= min_loc[0]+(w/2)
        y= min_loc[1]+(h/2)
        electrodes_loc += [(x,y)]
        
        #only for visualisation
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(image,top_left, bottom_right, 255, 2)
        cv2.circle(match_img, (int(x), int(y)),  20, (255,0,0), -1)

    return electrodes_loc, image

def find_electrodes(template, image):
    """
    finds the 12 points ressembling a electrode template the most and returns
    coordinates as a list as well as an image with the electrodes visualized with a rectancle
    """
    _, w, h = template.shape[::-1]
    electrodes_loc=[]
    #change image color to 1D grey
    template=color.rgb2grey(template)
    match_img=np.copy(image)
    match_img=color.rgb2grey(match_img)

    #iterate 12 times for 12 electrodes
    for i in range(0,12):
        #match to template
        match= cv2.matchTemplate(match_img, template, cv2.TM_CCOEFF_NORMED)
        #get min and max values and localisation
        min_val, max_val, min_loc, max_loc=cv2.minMaxLoc(match)
       
        #impute center of template rectangle and therefore coordinates
        x= max_loc[0]+(w/2)
        y= max_loc[1]+(h/2)
        electrodes_loc += [(x,y)]
        
        #only for visualisation
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(image,top_left, bottom_right, 255, 2)
        #black out the already detected electrode
        cv2.circle(match_img, (int(x), int(y)),  20, (0,0,0), -1)

    return electrodes_loc, image


def gray_im(img):
    if (len(img.shape) == 3):
        image=rgb2gray(img)
    else:
        image=img
    return image

def rgb2gray(img):
    return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def visual_callback_2d(background, fig=None):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_chan_vese` for visualizing the evolution
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
  
    # Callback for visual plotting
    callback = visual_callback_2d(image)

    # Morphological Chan-Vese 
    ima=ms.morphological_chan_vese(image, 35,
                               smoothing=3, lambda1=1, lambda2=1,
                               iter_callback=callback)

    return ima





def find_electrodes2(image, thresh_img):
    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the relevant components
    thresh_img = np.uint8(thresh_img)
    labels = measure.label(thresh_img, neighbors=8, background=0)
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(thresh_img.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        # if the number of pixels in the component is not too large, add it to mask
        if numPixels < 4500:
            mask = cv2.add(mask, labelMask)
        
    # find the contours in the mask, then sort them from left to
    # right
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    coordinates=[]
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)
        coordinates += [(int(cX), int(cY))]
   
    return image, coordinates

def calculateDistance(point1, point2):
    dist = np.sqrt((point1[1] - point2[1])**2 + (point1[0] - point2[0])**2)
    return dist
    
def ennumerate(loc, center, output_img):
    """
    Parameters
    ----------
    locs : list of electrode coordinates
    center : center of spiral, tuple of coordinates
    output_img : image where to plot numbers on 

    Returns
    -------
    sorted_loc : list of coordinates sorted by distance to center

    """
    #find distances between center and points
    dist=[]
    for i in loc:
        new_dist=calculateDistance(center, i)
        dist += [new_dist]
    

    #get a sorted list of points, according to distance from center
    sorted_loc=[]
    for i in range(12):
        near = np.argmin(dist)    

        sorted_loc += [loc[near]]
        #set high to remove from argmin
        dist[near]=10000

    #enumerate electrodes output
    for i in range(12):
        cv2.putText(output_image, str(i+1), (int(sorted_loc[i][0])+30, int(sorted_loc[i][1]+20)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255),3)
        
    plt.imshow(output_img)
    plt.show()
    return sorted_loc


def dist_between_points(x1, y1, x2, y2):
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist

def spiral_direction(cnt_coords):
    cnt_x = [cnt_coords[i][0] for i in range(len(cnt_coords))]
    cnt_y = [cnt_coords[i][1] for i in range(len(cnt_coords))]
    mean_y = np.mean(cnt_y)
    furthest_right = cnt_coords[np.argmax(cnt_x)]
    if furthest_right[1] > mean_y:
        spiral_dir = 'right'
    else: 
        spiral_dir = 'left'
    return spiral_dir

def enumerate_electrodes(cnt_coords, center):
    #dist = dist_between_points(cnt_coords[0][0], cnt_coords[0][1], cnt_coords[1][0], cnt_coords[1][1])
    cnt_x = [cnt_coords[i][0] for i in range(len(cnt_coords))]
    cnt_y = [cnt_coords[i][1] for i in range(len(cnt_coords))]
    mean_x = np.mean(cnt_x)
    mean_y = np.mean(cnt_y)
    furthest_right = cnt_coords[np.argmax(cnt_x)]
    furthest_left = cnt_coords[np.argmin(cnt_x)]
    spiral_dir = spiral_direction(cnt_coords)
    dist_to_mean = []
    distances = np.empty([len(cnt_coords), len(cnt_coords)])
    elec_coords = []
    for (x, y) in cnt_coords:
        dist_to_mean.append(dist_between_points(mean_x, mean_y, x, y))
    for i in range(len(cnt_coords)):
        for j in range(len(cnt_coords)):
            distances[i, j] = dist_between_points(cnt_coords[i][0], cnt_coords[i][1], cnt_coords[j][0], cnt_coords[j][1])
    median_dist = np.median(distances)  
    elec1 = dist_to_mean.index(min(dist_to_mean))
    elec_coords.append(cnt_coords[elec1])
    del cnt_coords[elec1]
    r = dist_between_points(center[0], center[1], elec_coords[0][0], elec_coords[0][1])
    for i in range(len(cnt_coords)):
        next_elec_x = elec_coords[i][0] + r * np.sin(median_dist/r)
        next_elec_y = elec_coords[i][1] - r * (1-np.cos(median_dist/r))
        tree = spatial.KDTree(cnt_coords)
        next_elec_index = tree.query((next_elec_x, next_elec_y))
        next_elec_coord = cnt_coords[next_elec_index[1]]
        elec_coords.append(next_elec_coord)
        cnt_coords.remove(next_elec_coord)

    return elec_coords


def center_electrodes(xdat, ydat):  
"""
The function of determining the center based on the coordinates of the electrodes
"""
    dx = np.diff(xdat)
    dy = np.diff(ydat)

    dx_1=dx**2
    dy_1=dy**2
    sum_ar=dx_1+dy_1
    sqr_arr=sum_ar**0.5

    heading = np.unwrap(np.arctan2(dy,dx))
    dphi = np.diff( heading )

    nda_1=ndat-2

    r = sqr_arr[0:nda_1]*np.tan(math.pi/2-dphi)

    xc = xdat[0:nda_1] + r * np.cos( heading[0:nda_1] + math.pi/2)
    yc = ydat[0:nda_1] + r * np.sin( heading[0:nda_1] + math.pi/2)

    xc1 = np.mean( xc )
    yc1 = np.mean( yc )
    return xc1, yc1




