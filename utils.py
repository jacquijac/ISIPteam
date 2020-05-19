from scipy import signal
from skimage import io, feature, color, filters
import matplotlib.pyplot as plt
import cv2
import numpy as np

from skimage import draw

from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_fill_holes
from scipy.ndimage.morphology import morphological_gradient, distance_transform_edt
from skimage import morphology as morph







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
    eroded = binary_erosion(im, structure=np.ones((2,2)), iterations = 20) [20:,20:]
    eroded = 1- eroded
    dilated = binary_dilation(eroded, structure=np.ones((11,11)))
    boundary = np.clip(dilated.astype(np.int)-eroded.astype(np.int), 0, 1)
    dt = distance_transform_edt(np.logical_not(boundary))
    edges = 1-morphological_gradient(im, size =3)
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
    m1 = fit_line_slope((center, last_electrode))
    m2 = fit_line_slope((center, other_electrode))
    
    tan_angle= (m1-m2)/(1+(m1*m2))
    angle= np.degrees(np.arctan(tan_angle))
    
