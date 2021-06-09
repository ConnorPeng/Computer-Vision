import glob
import os
from typing import Tuple
import math
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean and the standard deviation of the dataset.

    Note: convert the image in grayscale and then scale to [0,1] before computing
    mean and standard deviation

    Hints: use StandardScalar (check import statement)

    Args:
    -   dir_name: the path of the root dir
    Returns:
    -   mean: mean value of the dataset (np.array containing a scalar value)
    -   std: standard deviation of th dataset (np.array containing a scalar value)
    """

    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################

    SS = StandardScaler()
    print(SS)
    for pic in glob.glob(f"{dir_name}/*/*/*.jpg"):
      im = Image.open(pic).convert("L")
      im = np.array(im)
      im = im.reshape(-1, 1) 
      im = im/ 255
      SS.partial_fit(im)
    
    mean = SS.mean_
    print(mean)
    var = SS.var_
    print(var)
    std = math.sqrt(var)
    print(std)

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
