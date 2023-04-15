from typing import List, Tuple

import cv2
import numpy as np

class Matcher:
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
    
        return self.sift.detectAndCompute(image, None)
    
    def __find_good_matches(
        self,
        des1: np.ndarray,
        des2: np.ndarray,
        threshold: float,
        KNN: bool
    ) -> Tuple[Tuple, Tuple]:
    
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
    
        kp1, des1 = self.__detect_and_compute_keypoints(image1)
        kp2, des2 = self.__detect_and_compute_keypoints(image2)
        good_matches = self.__find_good_matches(des1, des2, threshold, KNN)
        return kp1, kp2, good_matches
    
