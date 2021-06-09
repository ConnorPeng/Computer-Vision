import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv, hsv2rgb
from typing import Tuple
import skimage.color
import matplotlib.pyplot as plt


def quantizeRGB(origImg: np.ndarray, k: int) -> np.ndarray:
    """
    Quantize the RGB image along all 3 channels and assign values of the nearest cluster
    center to each pixel. Return the quantized image and cluster centers.

    NOTE: Use the sklearn.cluster.KMeans function for clustering and set random_state = 101
    
    Args:
        - origImg: Input RGB image with shape H x W x 3 and dtype "uint8"
        - k: Number of clusters

    Returns:
        - quantizedImg: Quantized image with shape H x W x 3 and dtype "uint8"
        - clusterCenterColors: k x 3 vector of hue centers
    """
    quantizedImg = np.zeros_like(origImg)
    ######################################################################################
    ## TODO: YOUR CODE GOES HERE                                                        ##
    ######################################################################################

    H = origImg.shape[0]
    W = origImg.shape[1]
    D = origImg.shape[2]
    image_arr = origImg.reshape((H*W, D))
    #print(image_arr)
    kmimag = KMeans(n_clusters= k, random_state= 101).fit(image_arr)
    closest = kmimag.predict(image_arr)
    clusterCenterColors1 = kmimag.cluster_centers_
    print(clusterCenterColors1)
    clusterCenterColors = np.zeros_like(clusterCenterColors1)
    new = clusterCenterColors1[0]
    #print(new)
    index2 = 0
    for new in clusterCenterColors:
        index = 0
        for i in new:
            clusterCenterColors[index2][index] = int(clusterCenterColors1[index2][index])
            index = index + 1
        index2 = index2 + 1

    temp = 0
    for i in range(H):
        for j in range(W):
            quantizedImg[i][j] = clusterCenterColors[closest[temp]]
            temp = temp + 1


    return quantizedImg.astype(int), clusterCenterColors

def quantizeHSV(origImg: np.ndarray, k: int) -> np.ndarray:
    """
    Convert the image to HSV, quantize the Hue channel and assign values of the nearest cluster
    center to each pixel. Return the quantized image and cluster centers.

    NOTE: Consider using skimage.color for colorspace conversion
    NOTE: Use the sklearn.cluster.KMeans function for clustering and set random_state = 101

    Args:
        - origImg: Input RGB image with shape H x W x 3 and dtype "uint8"
        - k: Number of clusters

    Returns:
        - quantizedImg: Quantized image with shape H x W x 3 and dtype "uint8"
        - clusterCenterHues: k x 1 vector of hue centers
    """
    
    ######################################################################################
    ## TODO: YOUR CODE GOES HERE                                                        ##
    ######################################################################################

    ######################################################################################
    ## YOUR CODE ENDS HERE                                                              ##
    ######################################################################################

    new_img = skimage.color.rgb2hsv(origImg)
    H = new_img.shape[0]
    W = new_img.shape[1]
    D = new_img.shape[2]
    image_arr = np.reshape(new_img, (H*W, D))
    image_arr = image_arr[:, 0].reshape((H*W, 1))
    quantizedImg = np.copy(new_img)
    kmimag = KMeans(n_clusters= k, random_state= 101).fit(image_arr)
    closest = kmimag.predict(image_arr)
    clusterCenterColors1 = kmimag.cluster_centers_
    print("ab")
    temp = 0
    for i in range(H):
        for j in range(W):
            quantizedImg[i][j][0] = clusterCenterColors1[closest[temp]]
            temp += 1
    print(quantizedImg)
    quantizedImg = skimage.color.hsv2rgb(quantizedImg) * 255
    quantizedImg = quantizedImg.astype(int)
    return quantizedImg, clusterCenterColors1
    

def computeQuantizationError(origImg: np.ndarray, quantizedImg: np.ndarray) -> int:
    """
    Calculate the quantization error by finding the sum of squared differences between
    the original and quantized images. Implement a vectorized version (using numpy) of
    this error metric.

    Args:
        - origImg: Original input RGB image with shape H x W x 3 and dtype "uint8"
        - quantizedImg: Image obtained post-quantization with shape H x W x 3 and dtype "uint8"

    Returns
        - error: Quantization error
    """
    ######################################################################################
    ## TODO: YOUR CODE GOES HERE                                                        ##
    ######################################################################################

    ######################################################################################
    ## YOUR CODE ENDS HERE                                                              ##
    ######################################################################################
    error = np.sum(np.square(origImg - quantizedImg))
    return error
