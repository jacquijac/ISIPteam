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

from imageio import imread
from matplotlib import pyplot as plt
from skimage.io import imread
import morphsnakes as ms #pip install morphsnakes
import scipy
from scipy import ndimage




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



