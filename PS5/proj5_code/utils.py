import numpy as np
import cv2
import matplotlib.pyplot as plt
import os.path as osp
from glob import glob
from random import shuffle
import torch
import torchvision.transforms as transforms
from pycm import ConfusionMatrix


def im2single(im):
    im = im.astype(np.float32) / 255
    return im


def single2im(im):
    im *= 255
    im = im.astype(np.uint8)
    return im


def load_image(path):
    return im2single(cv2.imread(path))[:, :, ::-1]


def load_image_gray(path):
    img = load_image(path)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def rgb2gray(img: np.ndarray) -> np.ndarray:
    """ Use the coefficients used in OpenCV, found here:
            https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html

        Args:
        -   Numpy array of shape (M,N,3) representing RGB image

        Returns:
        -   Numpy array of shape (M,N) representing grayscale image
    """
    # Grayscale coefficients
    c = [0.299, 0.587, 0.114]
    return img[:, :, 0] * c[0] + img[:, :, 1] * c[1] + img[:, :, 2] * c[2]


def load_image_gray_tensor(path: str) -> torch.float:
    """
    Args:
    - path: string representing a file path to an image

    Returns:
    - float tensor of shape (m,n) and in range [0,1],
      representing a image in gray scale
    """

    gray_img = load_image_gray(path)
    tensor_type = torch.FloatTensor
    torch.set_default_tensor_type(tensor_type)
    to_tensor = transforms.ToTensor()
    gray_img_tensor = to_tensor(gray_img).unsqueeze(0)
    return gray_img_tensor


def arrayToTensor(img: np.ndarray) -> torch.float:
    tensor_type = torch.FloatTensor
    torch.set_default_tensor_type(tensor_type)
    to_tensor = transforms.ToTensor()
    gray_img_tensor = to_tensor(img).unsqueeze(0)
    return gray_img_tensor

def get_tiny_images(images_array, size=16):
    """
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images: a large dataset for non-parametric object and
    scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    This method resizes the images to given size, by default 16x16. Making
    the tiny images zero mean and unit length (normalizing them) will
    increase performance modestly.

    Args:
    -   image_arrays: list of N elements containing image in Numpy array, in
    grayscale
    -   size: size to resize the image to, default 16x16

    Returns:
    -   feats: N x d numpy array of resized and then vectorized tiny images
                e.g. if the images are resized to 16x16, d would be 256
    """

    images = [cv2.resize(img, (size, size))
              for img in images_array]
    feats = np.asarray([image.flatten() for image in images])

    mean = np.mean(feats, axis=1)
    feats = feats.reshape((feats.shape[1], feats.shape[0]))
    feats = np.array([feat - mean for feat in feats])
    feats = feats.reshape((feats.shape[1], feats.shape[0]))
    feats = np.array([feat / np.linalg.norm(feat) for feat in feats])

    return feats


def get_image_paths(data_path, categories, num_train_per_cat=100, fmt='jpg'):
    """
    This function returns lists containing the file path for each train
    and test image, as well as lists with the label of each train and
    test image. By default all four of these arrays will have 1500
    elements where each element is a string.
    :param data_path: path to the 'test' and 'train' directories
    :param categories: list of category names
    :param num_train_per_cat: max number of training images to use (per category)
    :param fmt: file extension of the images
    :return: lists: train_image_paths, test_image_paths, train_labels, test_labels
    """
    train_image_paths = []
    test_image_paths = []
    train_labels = []
    test_labels = []

    for cat in categories:
        # train
        pth = osp.join(data_path, 'train', cat, '*.{:s}'.format(fmt))
        pth = glob(pth)
        shuffle(pth)
        pth = pth[:num_train_per_cat]
        train_image_paths.extend(pth)
        train_labels.extend([cat] * len(pth))

        # test
        pth = osp.join(data_path, 'test', cat, '*.{:s}'.format(fmt))
        pth = glob(pth)
        shuffle(pth)
        pth = pth[:num_train_per_cat]
        test_image_paths.extend(pth)
        test_labels.extend([cat] * len(pth))

    return train_image_paths, test_image_paths, train_labels, test_labels


def get_image_arrays(data_path, categories, num_train_per_cat=100, fmt='jpg'):
    """
    This function returns lists containing the np array for each train
    and test image, as well as listss with the label of each train and
    test image. By default all four of these arrays will have 1500
    elements where each element is a string.
    :param data_path: path to the 'test' and 'train' directories
    :param categories: list of category names
    :param num_train_per_cat: max number of training images to use (per category)
    :param fmt: file extension of the images
    :return: lists: train_image_arrays, test_image_arrays, train_labels, test_labels
    """
    train_image_paths = []
    test_image_paths = []
    train_labels = []
    test_labels = []

    for cat in categories:
        # train
        pth = osp.join(data_path, 'train', cat, '*.{:s}'.format(fmt))
        pth = glob(pth)
        shuffle(pth)
        pth = pth[:num_train_per_cat]
        train_image_paths.extend(pth)
        train_labels.extend([cat] * len(pth))

        # test
        pth = osp.join(data_path, 'test', cat, '*.{:s}'.format(fmt))
        pth = glob(pth)
        shuffle(pth)
        pth = pth[:num_train_per_cat]
        test_image_paths.extend(pth)
        test_labels.extend([cat] * len(pth))

    train_image_arrays = [load_image_gray(p) for p in train_image_paths]
    test_image_arrays = [load_image_gray(p) for p in test_image_paths]

    return train_image_arrays, test_image_arrays, train_labels, test_labels


def show_results(test_labels,
                 categories,
                 abbr_categories,
                 predicted_categories,
                 savefile='../cm.png'):
    """
    shows the results
    :param test_image_paths:
    :param train_labels:
    :param test_labels:
    :param categories:
    :param abbr_categories:
    :param predicted_categories:
    :return:
    """
    cat2idx = {cat: idx for idx, cat in enumerate(categories)}

    # confusion matrix
    y_true = [cat2idx[cat] for cat in test_labels]
    y_pred = [cat2idx[cat] for cat in predicted_categories]
    cm = ConfusionMatrix(y_true, y_pred)
    plt.figure()
    plt_cm = []
    for i in cm.classes:
        row = []
        for j in cm.classes:
            row.append(cm.table[i][j])
        plt_cm.append(row)

    plt_cm = np.array(plt_cm)
    acc = np.sum(np.diag(plt_cm))/len(test_labels) * 100
    plt_cm = plt_cm.astype('float') / plt_cm.sum(axis=1)[:, np.newaxis]
    # print(plt_cm)
    plt.imshow(plt_cm, interpolation='nearest', cmap=plt.cm.get_cmap('jet'))
    plt.title('Confusion matrix. Accuracy = {:4.2f}%'.format(acc))
    tick_marks = np.arange(len(categories))
    plt.tight_layout()
    plt.xticks(tick_marks, abbr_categories, rotation=45)
    plt.yticks(tick_marks, categories)
    plt.savefig(savefile)

def generate_sample_points(height, width, stride):
    ys = np.arange(10, height - 10, stride)
    xs = np.arange(10, width - 10, stride)
    xv, yv = np.meshgrid(xs, ys)
    return xv.flatten(), yv.flatten()

