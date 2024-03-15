from PIL import Image, ImageDraw  # pillow package
import numpy as np
from scipy import ndimage

sigma = 1.2
low_thr = 100
high_thr = 150
num_lines = 10

def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr


def save_array_as_img(arr, file):
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min) / (max - min) * 255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)


def show_array_as_img(arr):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
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

def sobel(arr):
    '''Apply sobel operator on arr and return the result.'''
    # TODO: Please complete this function.
    # your code here
    sobel_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Gx = ndimage.convolve(arr, sobel_kernel_x)
    Gy = ndimage.convolve(arr, sobel_kernel_y)
    G = np.sqrt(np.square(Gx) + np.square(Gy))

    print("Sobel done, Saving...")

    return G, Gx, Gy


def nonmax_suppress(G, Gx, Gy):
    '''Suppress non-max value along direction perpendicular to the edge.'''
    assert G.shape == Gx.shape
    assert G.shape == Gy.shape
    # TODO: Please complete this function.
    # your code here
    padding_G = np.pad(G, mode='edge', pad_width=1)
    suppressed_G = np.copy(G)
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    for i in range(Gx.shape[0]):
        for j in range(G.shape[1]):
            if -22.5 < theta[i][j] < 22.5 or 157.5 < theta[i][j] < 180 or -180 < theta[i][j] < -157.5:
                n1 = padding_G[i + 1][j + 2]
                n2 = padding_G[i + 1][j]
            if 22.5 < theta[i][j] < 67.5 or -112.5 > theta[i][j] > -157.5:
                n1 = padding_G[i + 2][j + 2]
                n2 = padding_G[i][j]
            if 67.5 < theta[i][j] < 112.5 or -112.5 < theta[i][j] < -67.5:
                n1 = padding_G[i + 2][j + 1]
                n2 = padding_G[i][j + 1]
            if 112.5 < theta[i][j] < 157.5 or -67.5 < theta[i][j] < -22.5:
                n1 = padding_G[i][j + 2]
                n2 = padding_G[i + 2][j]
            if G[i][j] < n1 or G[i][j] < n2:
                suppressed_G[i][j] = 0

            # if -22.5<theta[i][j]<22.5 or 157.5<theta[i][j]<180 or -180<theta[i][j]<-157.5 :
            #     n1=G[i][j+1]
            #     n2=G[i][j-1]
            # if 22.5<theta[i][j]<67.5 or -112.5>theta[i][j]>-157.5 :
            #     n1=G[i+1][j+1]
            #     n2=G[i-1][j-1]
            # if 67.5<theta[i][j]<112.5 or -112.5<theta[i][j]<-67.5 :
            #     n1=G[i+1][j]
            #     n2=G[i-1][j]
            # if 112.5<theta[i][j]<157.5 or -67.5<theta[i][j]<-22.5 :
            #     n1=G[i-1][j+1]
            #     n2=G[i+1][j-1]
            # if G[i][j]<n1 or G[i][j]<n2:
            #     suppressed_G[i][j]=0
            print(i, ',', j, 'done')

        print("no_max done, saving")

    return suppressed_G


def thresholding(G, t):
    '''Binarize G according threshold t'''
    G_binary = G.copy()
    G_binary[G_binary <= t] = 0

    return G_binary


def hysteresis_thresholding(G, low, high):
    '''Binarize G according threshold low and high'''
    # TODO: Please complete this function.
    # your code here
    G_low = thresholding(G, low)
    G_low[G_low != 0] = 128
    G_high = thresholding(G, high)
    G_high[G_high != 0] = 255

    G_hyst = np.copy(G_high)
    visited = np.zeros_like(G)
    x, y = G_hyst.shape

    def searching_connection(i, j):
        if i >= x or i < 0 or j >= y or j < 0 or visited[i, j] == 1 or G_low[i, j] == 0:
            return
        visited[i, j] = 1
        if G_low[i][j] == 128:
            G_hyst[i, j] = 255
            searching_connection(i - 1, j - 1)
            searching_connection(i, j - 1)
            searching_connection(i + 1, j - 1)
            searching_connection(i - 1, j)
            searching_connection(i + 1, j)
            searching_connection(i - 1, j + 1)
            searching_connection(i, j + 1)
            searching_connection(i + 1, j + 1)

    for i in range(x):
        for j in range(y):
            if visited[i, j] == 1:
                continue
            if G_hyst[i, j] == 255:
                searching_connection(i, j)
            elif G_low[i, j] == 0:
                visited[i, j] = 1

    # for i in range(G_low.shape[0]):
    #     for j in range(G_low.shape[1]):
    #         if G_low[i][j]==128 && G_high[i][j]!=255 :
    #             check_kernel=G_high[i-1:i+2,j-1:j+2]
    #             if np.isin(np.array([5]), check_kernel).any():
    #                 G_hyst[i][j]=255
    #             else: G_hyst[i][j] =0
    print("Hysteresis_thresholding Done, saving...")
    return G_low, G_high, G_hyst


def hough(G_hyst):
    '''Return Hough transform of G'''
    # TODO: Please complete this function.
    # your code here
    # Define the Hough space size based on the maximum distance and angle
    max_distance = int(
        np.ceil(np.sqrt(G_hyst.shape[0] ** 2 + G_hyst.shape[1] ** 2)))  # Maximum distance from the origin
    max_angle = 180  # Maximum angle is usually 180 degrees

    # Create the Hough accumulator array
    accumulator = np.zeros((2 * max_distance, max_angle))

    # Iterate over the edge map to find edge pixels
    edge_indices = np.argwhere(G_hyst != 0)
    for y, x in edge_indices:
        for theta in range(max_angle):
            angle = np.deg2rad(theta)
            rho = int(x * np.cos(angle) + y * np.sin(angle)) + max_distance
            accumulator[rho, theta] += 1

    return accumulator


def get_lines_from_hough(accumulator, num_lines):
    lines = []
    # Find the indices of the highest-voting cells in the accumulator
    indices = np.argpartition(accumulator, -num_lines, axis=None)[-num_lines:]
    # Convert the indices to (rho, theta) values
    rhos, thetas = np.unravel_index(indices, accumulator.shape)
    for rho, theta in zip(rhos, thetas):
        # Convert back to original rho and theta values
        rho -= int(accumulator.shape[0] / 2)
        theta = np.deg2rad(theta)
        lines.append((rho, theta))
    return lines


def draw_lines(image, lines):
    draw = ImageDraw.Draw(image)
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = int(a * rho)
        y0 = int(b * rho)
        # Extend the line segment to be visible on the image
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        draw.line((x1, y1, x2, y2), fill=128, width=3)
    return draw


if __name__ == '__main__':
    input_path = '../data/road.jpeg'
    img = read_img_as_array(input_path)
    # show_array_as_img(img)
    # TODO: finish assignment Part II: detect edges on 'img'

    # 2.1 Gray-scale
    gray = rgb2gray(img)
    save_array_as_img(gray, '../data/2.1_gray.jpg')

    # 2.2 Gauss
    gauss = ndimage.gaussian_filter(gray, sigma=sigma, )
    save_array_as_img(gauss, '../data/2.2_gause.jpg')

    # 2.3 Sobel
    G, Gx, Gy = sobel(gauss)
    save_array_as_img(Gx, '../data/2.3_G_x.jpg')
    save_array_as_img(Gy, '../data/2.3_G_y.jpg')
    save_array_as_img(G, '../data/2.3_G.jpg')

    # 2.4 Non-max
    suppressed_G = nonmax_suppress(G, Gx, Gy)
    save_array_as_img(suppressed_G,'../data/2.4_supress.jpg')

    # 2.5 thresholding
    G_low, G_high, G_hyst = hysteresis_thresholding(suppressed_G, low_thr, high_thr)
    save_array_as_img(G_low, '../data/2.5_edgemap_low.jpg')
    save_array_as_img(G_high, '../data/2.5_edgemap_high.jpg')
    save_array_as_img(G_hyst, '../data/2.5_edgemap.jpg')

    # 2.6
    G_hough = hough(G_hyst)

    save_array_as_img(G_hough, '../data/2.6_hough.jpg')

    # 2.7
    img = Image.open(input_path)
    possible_lines = get_lines_from_hough(G_hough, num_lines)
    drew_img = draw_lines(img,possible_lines)
    img.save('../data/2.7_detection_result.jpg')
