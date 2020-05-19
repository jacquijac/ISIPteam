#import modules
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import feature, color, filters
from scipy import ndimage
from PIL import Image
import cv2
from utils import

# import pre and post images
#post = plt.imread("DATA/ID03/ID03post.png")
#pre = plt.imread("DATA/ID03/ID03pre.png")
#post = plt.imread("DATA/ID04/ID04post.png")
#pre = plt.imread("DATA/ID04/ID04pre.png")
#post = plt.imread("DATA/ID05/ID05post.png")
#pre = plt.imread("DATA/ID05/ID05pre.png")
#post = plt.imread("DATA/ID06/ID06post.png")
#pre = plt.imread("DATA/ID06/ID06pre.png")
#post = plt.imread("DATA/ID07/ID07post.png")
#pre = plt.imread("DATA/ID07/ID07pre.png")
#post = plt.imread("DATA/ID14/ID14post.png")
#pre = plt.imread("DATA/ID14/ID14pre.png")
#post = plt.imread("DATA/ID15/ID15post.png")
#pre = plt.imread("DATA/ID15/ID15pre.png")
#post = plt.imread("DATA/ID17/ID17post.png")
#pre = plt.imread("DATA/ID17/ID17pre.png")
#post = plt.imread("DATA/ID18/ID18post.png")
#pre = plt.imread("DATA/ID18/ID18pre.png")
#post = plt.imread("DATA/ID37/ID37post.png")
#pre = plt.imread("DATA/ID37/ID37pre.png")
#post = plt.imread("DATA/ID38/ID38post.png")
#pre = plt.imread("DATA/ID38/ID38pre.png")
post = plt.imread("DATA/ID55/ID55post.png")
pre = plt.imread("DATA/ID55/ID55pre.png")



#crop and normalize pre and post images
pre_crop = crop_image(pre)
post_crop = crop_image(post)
post_norm = normalize(post_crop)


def create_mask(img):
    '''
    Create a mask for lightest parts of post image, which containes the electrodes. 
    Mask is white for electrodes and black everywhere else.
    '''
    #initialize mask with zeros
    mask = np.zeros_like(img)
    #find mean pixel value
    mean_pixel = np.mean(img)
    #set mask=1 where pixel vales are greater than mean*1.75
    mask[np.where(np.logical_and(img>=mean_pixel*1.75,img<=1))] = 1
    #convert mask so it can be used in later functions 
    mask = np.uint8(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask 

def find_components(mask):
    '''
    Function finds all connected components (white parts of image), then determines which section of the image
    contains the most components. The components that are not in this section are blacked out. 
    '''
    #create empty list to store number of components in each section
    num_comps = list()
    #loop over mask, in increments of 50
    for i in range(0, mask.shape[0], 50):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask[:,i:i+400], connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #remove the background component
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        num_comps.append(nb_components)

    #find index of section with most components
    most_comps = num_comps.index(max(num_comps))
    #index corresponds to (index*50)+400 on image
    #black out everything except that section 
    mask[:, 0:most_comps*50] = 0
    mask[:, (most_comps*50)+400:] = 0


    #now get stats for only this section  
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    #we only want to keep the smallest components since they tend to form a circle around the spiral center
    #take maximum size to be (4/3)*mean(size)
    max_size = np.mean(sizes)*1.33

    #initialize output image with zeros 
    center = np.zeros((output.shape))
    
    #for every component in the image, you keep it only if it's below max_size
    for i in range(0, nb_components):
        if sizes[i] <= max_size:
            center[output == i + 1] = 255
            
    return center


def find_center(comp_img):
    '''
    Get x and y coordinates of all remaining components and find means of coordinates.
    '''
    y,x = np.nonzero(comp_img)

    #find mean of x and y, which represents the center of the electrodes circling the spiral center
    xmean = x.mean()
    ymean = y.mean()
    
    return xmean, ymean

mask = create_mask(post_norm)
comp_img = find_components(mask)
xmean, ymean = find_center(comp_img)

#plot pre image with red dot at spiral center
plt.plot(xmean, ymean, 'r.', markersize=14)
plt.imshow(pre_crop)

#import images - maybe loop throught



#define spiral center



#localize electrodes



#ennumerate electrodes




#calculate insertion angle



#output




