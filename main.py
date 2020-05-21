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
import utils

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
pre_crop = utils.crop_image(pre)
post_crop = utils.crop_image(post)
post_norm = utils.normalize(post_crop)



#find spiral center
mask = utils.create_mask(post_norm)
comp_img = utils.find_components(mask)
xmean, ymean = utils.find_center(comp_img)

#plot pre image with red dot at spiral center
plt.plot(xmean, ymean, 'r.', markersize=14)
plt.imshow(pre_crop)






#localize electrodes:
template = plt.imread("./DATA/ID55/ID55post.png")
template=(template[370:450, 250:330])

plt.imshow(find_electrodes(template, post_norm)[1])
plt.show


#ennumerate electrodes
#always 12 electrodes, always equally spaced
#if output as list can be used to loop over all electrodes




#calculate insertion angle
for i in electrodes:
  angle= (find_insertion_angle(center, i, i+1))
  #add output to excel part



#output




