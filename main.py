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

#maybe add loop?

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



#find spiral center
mask = create_mask(post_norm)
comp_img = find_components(mask)
xmean, ymean = find_center(comp_img)

#plot pre image with red dot at spiral center
plt.plot(xmean, ymean, 'r.', markersize=14)
plt.imshow(pre_crop)






#localize electrodes



#ennumerate electrodes




#calculate insertion angle



#output




