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

image = utils.gray_im(img)
call= utils.visual_callback_2d(image)
binar = utils.sharpShape(image, call)
#binar_float = np.uint8(binar)
#slicecanny = cv2.Canny(binar_float,0,1)
#plt.imshow(binar)
#distance_map = ndimage.distance_transform_edt(1- binar)
#plt.imshow(distance_map)



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

loc_electrodes, electrodes_image = utils.find_electrodes(template,post_norm)

plt.imshow(electrodes_image)
plt.show



#ennumerate electrodes
#always 12 electrodes, always equally spaced
#if output as list can be used to loop over all electrodes
enum_electrodes= utils.ennumerate(loc_electrodes, (xmean,ymean), post_norm)



#calculate insertion angle
for i in enum_electrodes:
  angle= (utils.find_insertion_angle(center, i, i+1))
  #add output to excel part



#output




