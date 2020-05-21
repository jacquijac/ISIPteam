from scipy import signal
from skimage import io, feature, color, filters
import matplotlib.pyplot as plt
import cv2
import numpy as np

from skimage import draw

from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_fill_holes
from scipy.ndimage.morphology import morphological_gradient, distance_transform_edt
from skimage import morphology as morph
import scipy.fftpack as fp
import numpy.fft






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




#It doesn't work perfectly yet, each picture has slightly different brightness parameters.
def brightPoints(image):
    img = cv2.imread(image) 
    hsv = cv2.cvtColor(img, cv.COLOR_BGR2HSV) 
    h, s, v = cv2.split(hsv) 
    limit = v.max () 

    hsv_min = np.array((0, 0, 180), np.uint8) 
    hsv_max = np.array((225, 225, limit), np.uint8)

    img1 = cv2.inRange(hsv, hsv_min, hsv_max) 

    moments = cv2.moments(img1, 1) 

    x_moment = moments['m01']
    y_moment = moments['m00']

    area = moments['m00']

    x = int(x_moment / area) 
    y = int(y_moment / area) 
    
    points = cv2.imwrite("points.jpg" , img1)
    
    return points


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

def find_electrodes(template, image):
    image=image.astype(np.float32)
    _, w, h = template.shape[::-1]
    electrodes_loc=[]
    template=color.rgb2grey(template)
    match_img=np.copy(image)
    match_img=color.rgb2grey(match_img)
    print(match_img.dtype)

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
