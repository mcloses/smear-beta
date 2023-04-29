from typing import List, Tuple

import cv2
import numpy as np

class HoldMatcher:
    """
    HoldMatcher class to match SIFT keypoints between two hold images.
    """
    def __init__(self):
        
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params,search_params)
        
    def __detect_and_compute_keypoints(
        self,
        image: np.ndarray,
    ) -> Tuple[cv2.KeyPoint, np.ndarray]:
        """
        Detect and compute SIFT keypoints and descriptors for a given image.

        :param image: An image.
        :type image: np.ndarray
        :return: Detected keypoints and their descriptors.
        :rtype: Tuple[cv2.KeyPoint, np.ndarray]
        """
        return self.sift.detectAndCompute(image, None)
    
    def __find_good_matches(
        self,
        des1: np.ndarray,
        des2: np.ndarray,
        threshold: float,
        KNN: bool
    ) -> Tuple[Tuple, Tuple]:
        """"
        Find good matches between two descriptors.

        :param des1: Descriptors from the first image.
        :type des1: np.ndarray
        :param des2: Descriptors from the second image.
        :type des2: np.ndarray
        :param threshold: Minimum distance between descriptors.
        :type threshold: float
        :param KNN: Use K-Nearest Neighbor matching.
        :type KNN: bool
        :return: Good matches between two images.
        :rtype: Tuple[Tuple, Tuple]
        """
        if KNN:
            matches = self.flann.knnMatch(des1, des2, k=2)
        else: matches = self.flann.BFMatcher(des1, des2)
        return [[m] for m, n in matches if m.distance < threshold*n.distance]
    
    def match(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: float = 0.75,
        KNN: bool = True
    ) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], Tuple[Tuple, Tuple]]:
        """
        Match SIFT keypoints between two images.

        :param image1: An image.
        :type image1: np.ndarray
        :param image2: An image.
        :type image2: np.ndarray
        :param threshold: Minimum distance between descriptors.
        :type threshold: float
        :param KNN: Use K-Nearest Neighbor matching.
        :type KNN: bool
        :return: Keypoints detected in the first and second image, and the good matches between them.
        :rtype: Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], Tuple[Tuple, Tuple]]
        """
        kp1, des1 = self.__detect_and_compute_keypoints(image1)
        kp2, des2 = self.__detect_and_compute_keypoints(image2)
        good_matches = self.__find_good_matches(des1, des2, threshold, KNN)
        return kp1, kp2, good_matches
    