import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from typing import Tuple
from scipy import misc, ndimage
from skimage import feature
from skimage.color import rgb2gray
from matplotlib.patches import Circle
from scipy import ndimage


def showCircles(
    img: np.ndarray,
    circles: np.ndarray,
    houghAccumulator: np.ndarray,
    showCenter: bool = False,
) -> None:
    """
    Function to plot the identified circles
    and associated centers in the input image.

    Args:
        - img: Input RGB image with shape H x W x 3 and dtype "uint8"
        - circles: An N x 3 numpy array containing the (x, y, radius)
            parameters associated with the detected circles
        - houghAccumulator: Accumulator array of size H x W
        - showCenter: Flag specifying whether to visualize the center
            or not
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

    ax1.set_aspect("equal")
    ax1.imshow(img)
    ax2.imshow(houghAccumulator)

    for circle in circles:
        x, y, rad = circle
        circ = Circle((y, x), rad, color="black", fill=False, linewidth=1.5)
        ax1.add_patch(circ)
        if showCenter:
            ax1.scatter(y, x, color="black")
    plt.show()


def detectCircles(
    img: np.ndarray, radius: int, threshold: float, useGradient: bool = False
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Implement a hough transform based circle detector that takes as input an
    image, a fixed radius, voting threshold and returns the centers of any detected
    circles of about that size and the hough space used for finding centers.

    NOTE: You are not allowed to use any existing hough transform detector function and
        are expected to implement the circle detection algorithm from scratch. As helper
        functions, you may use 
            - skimage.color.rgb2gray (for RGB to Grayscale conversion)
            - skimage.feature.canny (for edge detection)
            - denoising functions (if required)
        Additionally, you can use the showCircles function defined above to visualize
        the detected circles and the accumulator array.

    NOTE: You may have to tune the "sigma" parameter associated with your edge detector 
        to be able to detect the circles. For debugging, considering visualizing the
        intermediate outputs of your edge detector as well.

    For debugging, you can use im1.jpg to verify your implementation. See if you are able
    to detect circles of radii [75, 90, 100, 150]. Note that your implementation
    will be evaluated on a different image. For the sake of simplicity, you can assume
    that the test image will have the same basic color scheme as the provided image. Any
    hyper-parameters you tune for im1.jpg should also be applicable for the test image.

    Args:
        - img: Input RGB image with shape H x W x 3 and dtype "uint8"
        - radius: Radius of circle to be detected
        - threshold: Post-processing threshold to determine circle parameters
            from the accumulator array
        - useGradient: Flag that allows the user to optionally exploit the
            gradient direction measured at edge points.

    Returns:
        - circles: An N x 3 numpy array containing the (x, y, radius)
            parameters associated with the detected circles
        - houghAccumulator: Accumulator array of size H x W

    """
    ######################################################################################
    ## TODO: YOUR CODE GOES HERE                                                        ##
    ######################################################################################
    '''
    H = img.shape[0]
    W = img.shape[1]
    gray = rgb2gray(img)
    houghAccumulator = np.zeros_like(gray)
    edge = feature.canny(gray)
    #print(edge)
    for index, point in np.ndenumerate(edge):
        if point != 0:
            for i in range(0,360,5):
                degree = np.radians(i)
                a = int(int(index[0] - (radius * np.cos(degree))))
                b = int(int(index[1] + (radius * np.sin(degree))))
                if (b in range(edge.shape[0])) and (a in range(edge.shape[1])):
                    houghAccumulator[b,a] += 1
    #print("this is acc")
    #print(houghAccumulator)
    max_acc = np.max(houghAccumulator)
    set_threshold = max_acc * threshold
    filter_circle = np.nonzero(houghAccumulator >= set_threshold)
    filter_circle = np.transpose(filter_circle)
    #print(filter_circle)
    N = filter_circle.shape[0]
    circles = np.zeros((N, 3))
    #print(circles)
    circles[:,0] = filter_circle[:,1]
    circles[:,1] = filter_circle[:,0]
    circles[:,2] = int(radius)
    #print(circles)
    
    showCircles(img, circles,houghAccumulator, True)
    circles = circles[0:9, :]
        
    return circles, houghAccumulator
    '''

    #print(img)
    #print(radius)
    #print(threshold)
    #print(useGradient)

    H = img.shape[0]
    W = img.shape[1]
    gray = rgb2gray(img)
    dx = ndimage.sobel(gray, axis = 0)
    dy = ndimage.sobel(gray, axis = 1)
    edge = feature.canny(gray, sigma = 3)
    houghAccumulator = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            #check if the edge point is good
            check = edge[i][j]
            if check:
                if useGradient:
                    grad_theta = np.arctan2(dy[i][j], dx[i][j])
                    a = int(i - radius * np.cos(grad_theta))
                    b = int(j - radius * np.sin(grad_theta))
                    if (a >= 0 and a < H and b >= 0 and b < W):
                        houghAccumulator[a, b] += 1
                else:
                    for k in range(0,360,5):
                        degree = np.radians(k)
                        a = int(i - (radius * np.cos(degree)))
                        b = int(j + (radius * np.sin(degree)))
                        if (a >= 0 and a < H and b >= 0 and b < W):
                            houghAccumulator[a,b] += 1
    max_acc = np.max(houghAccumulator)
    set_threshold = max_acc * threshold
    circles = np.nonzero(houghAccumulator >= set_threshold)
    circles = np.transpose(circles)
    temp = np.full((circles.shape[0], 1), radius)
    circles = np.concatenate((circles, temp), axis = 1)
    print(circles)
    showCircles(img, circles,houghAccumulator, True)
    circles = circles[0:9, :]
    return circles, houghAccumulator

    
    ######################################################################################
    ## YOUR CODE ENDS HERE                                                              ##
    ######################################################################################
    