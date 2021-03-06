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


def threshold(image):
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
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)

    
    return image

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
