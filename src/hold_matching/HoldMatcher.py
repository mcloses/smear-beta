from typing import Dict, List, Tuple

import cv2
import numpy as np

from torch import TensorType

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
    
    def __SIFT_match(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: float,
        KNN: bool,
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
        :return: Keypoints detected in the first and second image, and the
                 good matches between them.
        :rtype: Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], Tuple[Tuple, Tuple]]
        """
        kp1, des1 = self.__detect_and_compute_keypoints(image1)
        kp2, des2 = self.__detect_and_compute_keypoints(image2)
        good_matches = self.__find_good_matches(des1, des2, threshold, KNN)
        return kp1, kp2, good_matches
    
    def match(
        self,
        origin_image: np.ndarray,
        dest_image: np.ndarray,
        origin_image_predictions: Dict[str, TensorType],
        dest_image_predictions: Dict[str, TensorType],
        selected_holds_id: List[int],
        sift_keypoint_match_score: int = 1,
        color_range_match_score: int = 10,
        color_range_degrees: int = 30,
        sift_knn_matcher: bool = True,
        threshold: float = 0.75,
    ) -> List[int]:
        """
        Match selected holds in the original image to
        holds in the destination image.

        :param origin_image: The original image containing the holds to match.
        :type origin_image: np.ndarray
        :param dest_image: The destination image to search for holds.
        :type dest_image: np.ndarray
        :param origin_image_predictions: Predictions for holds in the original
                                         image in Detectron2 format.
        :type origin_image_predictions: Dict[str, TensorType]
        :param dest_image_predictions: Predictions for holds in the destination
                                       image in Detectron2 format.
        :type dest_image_predictions: Dict[str, TensorType]
        :param selected_holds_id: List of ids for holds in the original image
                                  to match.
        :type selected_holds_id: List[int]
        :param sift_keypoint_match_score: The score to assign for each matched
                                          SIFT keypoint.
        :type sift_keypoint_match_score: int, default 1
        :param hue_color_range_match_score: The score to assign for a hue
                                            color range match.
        :type hue_color_range_match_score: int, default 10
        :param hue_color_range_degrees: The hue color range in degrees
                                        for a match.
        :type hue_color_range_degrees: int, default 30
        :param sift_knn_matcher: Whether to use the KNN matcher for SIFT
                                 keypoint matching. If False, use the 
                                 BFMatcher.
        :type sift_knn_matcher: bool, default True
        :param threshold: The minimum distance between descriptors.
        :type threshold: float, default 0.75
        :return: List of ids for the matched hold candidates
                 in the destination image.
        :rtype: List[int]
        """
        
        # initialize dictionary to hold scores for each hold
        dest_hold_scores = {
            index: 0 for index in range(
                len(dest_image_predictions["instances"])
            )
        }
        bb_margin = int(origin_image.shape[0] * 0.01)
        
        for hold_index in selected_holds_id:
            
            bbox = (
                origin_image_predictions["instances"][hold_index]
                .get("pred_boxes")
                .tensor
                .cpu()
                .numpy()[0]
            )

            # obtain smaller image around hold
            hold_area = origin_image[
                int(max(0, bbox[1]-bb_margin)):
                int(min(origin_image.shape[0], bbox[3]+bb_margin)),
                int(max(0, bbox[0]-bb_margin)):
                int(min(origin_image.shape[1], bbox[2]+bb_margin))
            ].copy()
            
            # find good matches between hold and dest image
            _, kp2, good_matches = self.__SIFT_match(
                hold_area,
                dest_image,
                threshold,
                sift_knn_matcher
            )
            matched_points = [
                kp2[match[0].trainIdx].pt
                for match in good_matches
            ]
            
            # if hold keypoints have been found in dest image
            if len(matched_points) > 0:
                # obtain hold color mean based on hold segmentation mask
                hold_col_mean = cv2.mean(
                        origin_image,
                        mask=(
                            origin_image_predictions["instances"][hold_index]
                            .get("pred_masks")
                            .cpu()
                            .numpy()[0] * 255
                            ).astype(np.uint8)
                )
                
                # loop through dest image hold predictions
                for index in range(len(dest_image_predictions["instances"])):
                    bbox = (
                        dest_image_predictions["instances"][index]
                        .get("pred_boxes")
                        .tensor
                        .cpu()
                        .numpy()[0]
                    )
                    # award points for each matched keypoint within the dest
                    # image hold candidate bbox
                    for point in matched_points:
                        if (
                            bbox[0] < point[0] < bbox[2] and
                            bbox[1] < point[1] < bbox[3]
                        ):
                            dest_hold_scores[index] += sift_keypoint_match_score
                    
                    # obtain hold candidate color hue mean based on hold
                    # segmentation mask
                    cand_col_mean = cv2.mean(
                        dest_image,
                        mask=(
                            dest_image_predictions["instances"][index]
                            .get("pred_masks")
                            .cpu()
                            .numpy()[0] * 255
                            ).astype(np.uint8)
                    )
                    # award points if candidate hold color is within
                    # range of hold color          
                    if (
                        bool(cv2.inRange(
                            np.array(cand_col_mean[:3]),
                            np.array(hold_col_mean[:3])-color_range_degrees,
                            np.array(hold_col_mean[:3])+color_range_degrees,
                        ).all())
                    ):
                        dest_hold_scores[index] += color_range_match_score
                        
        sorted__dest_hold_scores = sorted(
            dest_hold_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        # select top N hold candidates in dest image
        hold_candidates = [
            key for key, _ 
            in sorted__dest_hold_scores[
                :len(selected_holds_id)
            ]
        ]
        
        return hold_candidates
    