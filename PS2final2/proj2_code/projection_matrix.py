import time
from typing import Tuple

import numpy as np
from scipy.linalg import rq
from scipy.optimize import least_squares


def objective_func(x: np.ndarray, **kwargs):
    """
    Calculates the difference in image (pixel coordinates) and returns 
    it as a 2*n_points vector

    Args: 
    -        x: numpy array of 11 parameters of P in vector form 
                (remember you will have to fix P_34=1) to estimate the reprojection error
    - **kwargs: dictionary that contains the 2D and the 3D points. You will have to
                retrieve these 2D and 3D points and then use them to compute 
                the reprojection error.
    Returns:
    -     diff: A 2*N_points-d vector (1-D numpy array) of differences betwen 
                projected and actual 2D points

    """

    diff = None

    points_2d = kwargs['pts2d']
    points_3d = kwargs['pts3d']

    ##############################
    # TODO: Student code goes here
    print(points_2d)
    print(points_3d)
    #fix P_34 = 1
    x = np.append(x, 1.0)
    p_matrix = x.reshape((3, 4))
    print(p_matrix)
    new_projection = projection(p_matrix, points_3d)
    diff = new_projection - points_2d
    N_point = diff.shape[0]
    diff = np.reshape(diff, (2 * N_point, ))
    ##############################

    return diff


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
        Computes projection from [X,Y,Z,1] in non-homogenous coordinates to
        (x,y) in non-homogenous image coordinates.

        Args:
        -  P: 3x4 projection matrix
        -  points_3d : n x 3 array of points [X_i,Y_i,Z_i]

        Returns:
        - projected_points_2d : n x 2 array of points in non-homogenous image coordinates
    """

    projected_points_2d = None

    assert points_3d.shape[1]==3
    ##############################
    # TODO: Student code goes here
    print(points_3d)
    rows = points_3d.shape[0]
    #add ones to the matrix
    new_ones = np.ones((rows, 1))
    points_3d = np.append(points_3d, new_ones, axis = 1)
    points_3d = points_3d.T
    print(points_3d)
    two_d = np.dot(P, points_3d)
    two_d[0] = two_d[0]/two_d[2]
    two_d[1] = two_d[1]/two_d[2]
    projected_points_2d = two_d[0:2]    
    print(projected_points_2d)
    projected_points_2d = projected_points_2d.T
    ##############################

    return projected_points_2d


def estimate_camera_matrix(pts2d: np.ndarray,
                           pts3d: np.ndarray,
                           initial_guess: np.ndarray) -> np.ndarray:
    '''
        Calls least_squares form scipy.least_squares.optimize and
        returns an estimate for the camera projection matrix

        Args:
        - pts2d: n x 2 array of known points (x_i, y_i) in image coordinates 
        - pts3d: n x 3 array of known points in 3D, (X_i, Y_i, Z_i, 1) 
        - initial_guess: 3x4 projection matrix initial guess

        Returns:
        - P: 3x4 estimated projection matrix 

        Note: Because of the requirements of scipy.optimize.least_squares
              you will have to pass the projection matrix P as a vector.
              Since we will fix P_34 to 1 you will not need to pass all 12
              matrix parameters. 

              You will also have to put pts2d and pts3d into a kwargs dictionary
              that you will add as an argument to least squares.

              We recommend that in your call to least_squares you use
              - method='lm' for Levenberg-Marquardt
              - verbose=2 (to show optimization output from 'lm')
              - max_nfev=50000 maximum number of function evaluations
              - ftol \
              - gtol  --> convergence criteria
              - xtol /
              - kwargs -- dictionary with additional variables 
                          for the objective function
    '''

    P = None

    start_time = time.time()

    kwargs = {'pts2d': pts2d,
              'pts3d': pts3d}

        
    ##############################
    # TODO: Student code goes here
    #print(initial_guess)
    #as vector
    temp = initial_guess.flatten()
    P = least_squares(objective_func, temp[:11], method='lm', verbose=2, max_nfev=50000, kwargs = kwargs).x
    #fix P_34 = 1
    P = np.append(P, 1.0)
    print(P)
    P = P.reshape((3, 4))
    ##############################

    print("Time since optimization start", time.time() - start_time)

    return P


def decompose_camera_matrix(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
        Decomposes the camera matrix into the K intrinsic and R rotation matrix

        Args:
        -  P: 3x4 numpy array projection matrix

        Returns:

        - K: 3x3 intrinsic matrix (numpy array)
        - R: 3x3 orthonormal rotation matrix (numpy array)

        hint: use scipy.linalg.rq()
    '''
    K = None
    R = None

    ##############################
    # TODO: Student code goes here
    first_three = P[:, 0:3]
    K, R = rq(first_three)
    ##############################

    return K, R


def calculate_camera_center(P: np.ndarray,
                            K: np.ndarray,
                            R_T: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   P: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    cc = None

    ##############################
    # TODO: Student code goes here
    #print(R_T)
    #print(K)
    matrix_one = np.dot(K, R_T)
    matrix_one = np.linalg.inv(matrix_one)
    print(matrix_one)
    col = P.shape[1] - 1
    P = P[:, col].flatten()
    print(P)
    cc = np.dot(matrix_one, P)
    
    ##############################

    return -cc
