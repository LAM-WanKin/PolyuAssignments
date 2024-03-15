from PIL import Image  # pillow package
import numpy as np
from scipy import ndimage
import os


def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr


def save_array_as_img(arr, file):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min) / (max - min) * 255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)


def show_array_as_img(arr, rescale='minmax'):
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min) / (max - min) * 255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.show()


def rgb2gray(arr):
    R = arr[:, :, 0]  # red channel
    G = arr[:, :, 1]  # green channel
    B = arr[:, :, 2]  # blue channel
    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    return gray


#########################################
## Please complete following functions ##
#########################################
def sharpen(img, sigma, alpha):
    '''Sharpen the image. 'sigma' is the standard deviation of Gaussian filter. 'alpha' controls how much details to add.'''
    # TODO: Please complete this function.
    # your code here
    arr = np.copy(img)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    detail_r = r - ndimage.gaussian_filter(r, sigma)
    detail_g = g - ndimage.gaussian_filter(g, sigma)
    detail_b = b - ndimage.gaussian_filter(b, sigma)
    arr[:, :, 0] = np.clip(img[:, :, 0] + alpha * detail_r, 0, 255).astype(np.uint8)
    arr[:, :, 1] = np.clip(img[:, :, 1] + alpha * detail_g, 0, 255).astype(np.uint8)
    arr[:, :, 2] = np.clip(img[:, :, 2] + alpha * detail_b, 0, 255).astype(np.uint8)

    return arr


def median_filter(img, s):
    '''Perform median filter of size s x s to image 'arr', and return the filtered image.'''
    # TODO: Please complete this function.
    # your code here
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    out_img = np.zeros(img.shape)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            kernel_r = r[x:x + s, y:y + s]
            kernel_g = g[x:x + s, y:y + s]
            kernel_b = b[x:x + s, y:y + s]
            k_med_r = np.median(kernel_r)
            k_med_g = np.median(kernel_g)
            k_med_b = np.median(kernel_b)
            out_img[x][y][0] = k_med_r
            out_img[x][y][1] = k_med_g
            out_img[x][y][2] = k_med_b
            print("done for pixel:", x, y)
    arr = out_img

    return arr


if __name__ == '__main__':
    input_path = '../data/rain.jpeg'
    save_path = '../data/'
    img = read_img_as_array(input_path)
    # show_array_as_img(img)
    # TODO: finish assignment Part I.\

    ######### Sharpenning #########
    str_save = "../data/1.1_sharpened.jpg"
    sharpen_img = sharpen(img, 1, 2)
    show_array_as_img(sharpen_img)
    save_array_as_img(sharpen_img, str_save)
    ###############################

    # ######### median_filter #########
    # s = 7
    # str_save = "../data/1.2_derained.jpg"
    # de_rain_img = median_filter(img, s)
    # show_array_as_img(de_rain_img)
    # save_array_as_img(de_rain_img, str_save)
    # #################################
