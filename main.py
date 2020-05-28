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
import xlwt 
from xlwt import Workbook 

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


#assert last electrode

#find insertion angle
last=len(enum_electrodes)
ref = utils.find_insertion_angle(center, enum_electrodes[-1], enum_electrodes[-1])
print(last, 'New angle: {:.2f}'.format(abs(ref)), 'Total insertion angle: {:.2f}'.format(ref))

for i in range(len(enum_electrodes)-1):
    angle= (utils.find_insertion_angle(center, enum_electrodes[-i], enum_electrodes[-i+1]))
    ref+=abs(angle)
    number=11-i
    print(number, 'New angle: {:.2f}'.format(abs(angle)), 'Total insertion angle: {:.2f}'.format(ref))



#output

  
# Workbook is created 
wb = Workbook() 
  
# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('ISIP_output') 
  
sheet1.write(0, 0, 'Electrode i') 
sheet1.write(0, 1, 'Angle theta') 

for i in range(1,13):
    sheet1.write(i, 0, i) 
    sheet1.write(i, 1, output[i]) 


  
wb.save('ISIP_angles.xls')

#### Loop to input files ####

#create list of id numbers
ids = ["03", "04", "05", "06", "07", "14", "15", "17", "18", "37", "38", "55"]
#loop over all id numbers
for i in ids:
    #set pre and post images
    pre = plt.imread('DATA/ID' + i + '/ID' + i + 'pre.png')
    post = plt.imread('DATA/ID' + i + '/ID' + i + 'post.png')
    #crop and normalize pre and post images
    pre_crop = crop_image(pre)
    post_crop = crop_image(post)
    post_norm = normalize(post_crop)
    #find spiral center
    mask = utils.find_bright_points(post_norm)
    comp_img = utils.find_components(mask)
    x_center, y_center = utils.find_center(comp_img)
    center = (x_center, y_center)
    #plot pre image with red dot at spiral center
    plt.plot(x_center, y_center, 'r.', markersize=14)
    plt.imshow(pre_crop)
    plt.show()
    #localize electrodes 
    elec_img, elec_coords = utils.find_electrodes(post_norm, mask)
    #ennumerate electrodes 
    enum_electrodes = utils.ennumerate_electrodes(elec_coords) 
    ##depending on format of output of ennumerate function, add plot with electrodes labeled
    ##
    #find insertion angle
    last=len(enum_electrodes)
    ref = utils.find_insertion_angle(center, enum_electrodes[-1], enum_electrodes[-1])
    print('For ID' + i + 'angles are:')
    print(last, 'New angle: {:.2f}'.format(abs(ref)), 'Total insertion angle: {:.2f}'.format(ref))
    for i in range(len(enum_electrodes)-1):
        angle = (utils.find_insertion_angle(center, enum_electrodes[-i], enum_electrodes[-i+1]))
        ref += abs(angle)
        number = 11-i
        print(number, 'New angle: {:.2f}'.format(abs(angle)), 'Total insertion angle: {:.2f}'.format(ref))


