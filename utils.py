def crop_image(img):
    crop_img = img[0:img.shape[0], 50:img.shape[1]]
    return crop_img

def normalize(img):
    min_pixel = np.amin(img)
    max_pixel = np.amax(img)
    norm_img = img - min_pixel * 1 / (max_pixel - min_pixel)
    return norm_img
