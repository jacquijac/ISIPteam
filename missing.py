from skimage.feature import blob_log
from matplotlib import pyplot as plt
from math import sqrt
from skimage.color import rgb2gray


#finds electrodes for ID14, 15 and 17
def missing_electrodes(image):
    #input: image
    #output: coordinates of blobs
    im = rgb2gray(image)
    blobs_log = blob_log(im, max_sigma=30, num_sigma=20, threshold=.03) #optimal values
# Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)


    fig, ax = plt.subplots(1, 1)
    plt.imshow(im)
    for blob in blobs_log:
        y, x, r = blob
        c = plt.Circle((x, y), r+5, color='red', linewidth=2, fill=False)
        ax.add_patch(c)
        
    return blobs_log[:,:2].astype(int)