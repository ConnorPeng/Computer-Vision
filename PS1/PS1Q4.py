import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io

class Prob4():
    def __init__(self):
        """Load input color image indoor.png and outdoor.png here as class variables."""
        ###### START CODE HERE ######


        ###### END CODE HERE ######
        pass
    
    def prob_4_1(self):
        """Plot R,G,B channels separately and also their corresponding LAB space channels separately"""
        
        ###### START CODE HERE ######


        ###### END CODE HERE ######
        pass

    def prob_4_2(self):
        """
        Convert the loaded RGB image to HSV and return HSV matrix without using inbuilt functions. Return the HSV image as HSV. Make sure to use a 3 channeled RGB image with floating point values lying between 0 - 1 for the conversion to HSV.

        Returns:
            HSV image (3 channeled image with floating point values lying between 0 - 1 in each channel)
        """
        
        img = io.imread('inputPS1Q4.jpg') 
        img = img / 255.0
        
        ###### START CODE HERE ######


        ###### END CODE HERE ######
        pass
    
        ###### return HSV ######
        

        
if __name__ == '__main__':
    
    p4 = Prob4()
    
    p4.prob_4_1()

    HSV = p4.prob_4_2()





