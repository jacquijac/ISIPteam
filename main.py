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


  

#### Loop to input files ####

#create list of id numbers of all pictures
ids = ["03", "04", "05", "06", "07", "14", "15", "17", "18", "37", "38", "55"]

#loop over all id numbers
for i in ids:
    
    #set pre and post images
    pre = plt.imread('DATA/ID' + i + '/ID' + i + 'pre.png')
    post = plt.imread('DATA/ID' + i + '/ID' + i + 'post.png')
    
    #crop out text of images and normalize pre and post images to homogenize color scale
    pre_crop = utils.crop_image(pre)
    post_crop = utils.crop_image(post)
    post_norm = utils.normalize(post_crop)
    
    #find spiral center
    mask = utils.find_bright_points(post_norm)
    comp_img = utils.find_components(mask)
    x_center, y_center = utils.find_center(comp_img)
    center = (x_center, y_center)
    
    #plot pre-operative image with red dot at spiral center
    plt.plot(x_center, y_center, 'r.', markersize=14)
    plt.imshow(pre_crop)
    plt.show()
    
    #localize electrodes 
    elec_img, elec_coords = utils.find_electrodes2(post_norm, mask)
    
    #ennumerate electrodes 
    enum_electrodes = utils.ennumerate_electrodes(elec_coords) 
    ##depending on format of output of ennumerate function, add plot with electrodes labeled
    ##
    #find insertion angle
    last=len(enum_electrodes)
    #find individual angles between points
    ref = utils.find_insertion_angle(center, enum_electrodes[-1], enum_electrodes[-1])
    print('For ID' + i + 'angles are:')
    print(last, 'New angle: {:.2f}'.format(abs(ref)), 'Total insertion angle: {:.2f}'.format(ref))
    #add up angles for insertion depth
    output={12:ref}
    for j in range(len(enum_electrodes)-1):
        angle = (utils.find_insertion_angle(center, enum_electrodes[-j], enum_electrodes[-j+1]))
        ref += abs(angle)
        number = 11-j
        print(number, 'New angle: {:.2f}'.format(abs(angle)), 'Total insertion angle: {:.2f}'.format(ref))
        output[number]=ref

    #output to excel
    # Workbook is created 
    wb = Workbook() 
  
    # add_sheet is used to create sheet. 
    sheet1 = wb.add_sheet('ISIP_output') 
  
    #Write output to excel sheet
    sheet1.write(0, 0, 'Electrode i') 
    sheet1.write(0, 1, 'Angle theta')
    for k in range(1,13):
        sheet1.write(k, 0, ) 
        sheet1.write(k, 1, output[k]) 
    wb.save('ISIP_angles.xls')
