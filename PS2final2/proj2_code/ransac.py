import math
from typing import Tuple

import numpy as np

from proj2_code import fundamental_matrix, two_view_data
from proj2_code.least_squares_fundamental_matrix import solve_F


def calculate_num_ransac_iterations(prob_success: float, 
                                    sample_size: int, 
                                    ind_prob_correct: float) -> int:
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None

    ##############################
    # TODO: Student code goes here
    prob_fail = 1 - prob_success
    all_success = ind_prob_correct**sample_size
    num_samples = math.log(prob_fail)/math.log(1-all_success)
    num_samples = int(num_samples)
    ##############################

    return num_samples


def find_inliers(x_0s: np.ndarray, 
                 F: np.ndarray, 
                 x_1s: np.ndarray, 
                 threshold: float) -> np.ndarray:
    """ Find the inliers' indices for a given model.

    There are multiple methods you could use for calculating the error
    to determine your inliers vs outliers at each pass. However, we suggest
    using the magnitude of the line to point distance function we wrote for the
    optimization in part 2.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   F: The proposed fundamental matrix
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    -   threshold: the maximum error for a point correspondence to be
                    considered an inlier
    Each row in x_1s and x_0s is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -    inliers: 1D array of the indices of the inliers in x_0s and x_1s

    """

    inliers = np.array([])

    ##############################
    # TODO: Student code goes here
    zero_length = x_0s.shape[0]
    one_length = x_1s.shape[0]
    zero_column = np.ones((zero_length, 1))
    one_column = np.ones((one_length, 1))
    #check if the dimension is 2, add columns of ones
    if x_0s.shape[1] == 2:
        x_0s = np.append(x_0s, zero_column, axis = 1)
    if x_1s.shape[1] == 2:
        x_1s = np.append(x_1s, one_column, axis = 1)
    #use signed_point_line_errors to calculate error
    error = fundamental_matrix.signed_point_line_errors(x_0s, F, x_1s)
    #print(error)
    error_length = int(len(error)/2)
    print(error_length)
    for i in range(error_length):
        if(abs(error[2*i]) <= threshold and abs(error[2*i + 1]) <= threshold):
            inliers = np.append(inliers, i)
    ##############################
    return inliers


def ransac_fundamental_matrix(x_0s: int, 
                              x_1s: int) -> Tuple[
                                  np.ndarray, np.ndarray, np.ndarray]:
    """Find the fundamental matrix with RANSAC.

    Use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You will call your
    solve_F() from part 2 of this assignment
    and calculate_num_ransac_iterations().

    You will also need to define a new function (see above) for finding
    inliers after you have calculated F for a given sample.

    Tips:
        0. You will need to determine your P, k, and p values.
            What is an acceptable rate of success? How many points
            do you want to sample? What is your estimate of the correspondence
            accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for
            creating your random samples
        2. You will want to call your function for solving F with the random
            sample and then you will want to call your function for finding
            the inliers.
        3. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 1.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    Each row is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_x_0: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the left image that are inliers with
                   respect to best_F
    -   inliers_x_1: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the right image that are inliers with
                   respect to best_F

    """

    best_F = None
    inliers_x_0 = []
    inliers_x_1 = []

    ##############################
    # TODO: Student code goes here

    P = 0.99
    k = 9
    p = 0.90
    threshold = 1.0
    iterations = calculate_num_ransac_iterations(P, k, p)
    length_x = int(len(x_0s))
    #iterate
    for _ in range(iterations):
        #creating random samples
        rand_sample = np.random.choice(length_x, k)
        rand_x_0s = x_0s[rand_sample]
        rand_x_1s = x_1s[rand_sample]
        #create an initial F
        Fund = solve_F(rand_x_0s, rand_x_1s)
        #find inliers
        inliers = find_inliers(x_0s, Fund, x_1s, threshold)
        num_inlier = len(inliers)
        temp1 = len(inliers_x_0)
        temp2 = len(inliers_x_1)
        if(num_inlier > temp1 and num_inlier > temp2):
            best_F = Fund
            index = inliers.astype('int')
            inliers_x_0 = x_0s[index]
            inliers_x_1 = x_1s[index]

    ##############################

    return best_F, inliers_x_0, inliers_x_1
