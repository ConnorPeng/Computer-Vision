import numpy as np


def compute_feature_distances(features1, features2):
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)
    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    n = features1.shape[0]
    m = features2.shape[0]
    dists = np.zeros((n, m))
    for i in range(n):
          for j in range(m):
                dists[i][j] = np.linalg.norm(features1[i,:] - features2[j,:],ord=2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dists


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).
    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).
    You should call `compute_feature_distances()` in this function, and then
    process the output.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)
    - x1: A numpy array of shape (n,) containing the x-locations of features1
    - y1: A numpy array of shape (n,) containing the y-locations of features1
    - x2: A numpy array of shape (m,) containing the x-locations of features2
    - y2: A numpy array of shape (m,) containing the y-locations of features2
    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match
    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    #version 1

    matches = []
    confidences = []
    n = features1.shape[0]
    m = features2.shape[0]
    dists = compute_feature_distances(features1,features2)
    print(dists)
    for i in range(n):
        num = np.argpartition(dists[i], 2)[:2]
        num_m = num[0]
        num_n = num[1]
        first = dists[i][num_m]
        second = dists[i][num_n]
        #ratio test
        div = first/second
        if div <= 0.95:
            confidences.append(div)
            matches.append([i,num_m])
    matches = np.asarray(matches)
    confidences = np.asarray(confidences)
    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences

def pca(fvs1, fvs2, n_components= 24):
    """
    Perform PCA to reduce the number of dimensions in each feature vector resulting in a speed up.
    You will want to perform PCA on all the data together to obtain the same principle components.
    You will then resplit the data back into image1 and image2 features.

    Helpful functions: np.linalg.svd, np.mean, np.cov

    Args:
    -   fvs1: numpy nd-array of feature vectors with shape (k,128) for number of interest points 
        and feature vector dimension of image1
    -   fvs1: numpy nd-array of feature vectors with shape (m,128) for number of interest points 
        and feature vector dimension of image2
    -   n_components: m desired dimension of feature vector

    Return:
    -   reduced_fvs1: numpy nd-array of feature vectors with shape (k, m) with m being the desired dimension for image1
    -   reduced_fvs2: numpy nd-array of feature vectors with shape (k, m) with m being the desired dimension for image2
    """

    reduced_fvs1, reduced_fvs2 = None, None
    #############################################################################
    # TODO: YOUR PCA CODE HERE                                                  #
    #############################################################################
    k = fvs1.shape[0]
    m = fvs2.shape[0]
    num = fvs2.shape[1]
    mean_fvs = np.mean(np.vstack((fvs1,fvs2)), axis = 0)
    new_fvs = np.vstack((fvs1,fvs2)) - mean_fvs
    u, s, vh = np.linalg.svd(new_fvs)
    temp = vh[:n_components].T
    fvs =  np.dot(new_fvs, temp)
    
    reduced_fvs1 = fvs[:k]
    reduced_fvs2 = fvs[k:]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return reduced_fvs1, reduced_fvs2

def accelerated_matching(features1, features2, x1, y1, x2, y2):
    """
    This method should operate in the same way as the match_features function you already coded.
    Try to make any improvements to the matching algorithm that would speed it up.
    One suggestion is to use a space partitioning data structure like a kd-tree or some
    third party approximate nearest neighbor package to accelerate matching.
    Note that doing PCA here does not count. This implementation MUST be faster than PCA
    to get credit.
    """

    #############################################################################
    # TODO: YOUR CODE HERE                                                  #
    #############################################################################
    raise NotImplementedError('`accelerated_matching` function in ' +
    '`student_feature_matching.py` needs to be implemented')
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return matches, confidences