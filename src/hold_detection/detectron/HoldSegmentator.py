import os
import pickle

from typing import Dict, Tuple

from detectron2.engine import DefaultPredictor
from numpy import ndarray
import torch

from smear_beta_utils.yolo_utils import letterbox


class HoldSegmentator:
    """
    Instance segmentator class
    """
    def __init__(
        self,
        model_cfg_path : str,
        model_path : str,
        confidence_threshold : float,
        nms_threshold : float
    ):
        """
        Hold segmentator class
        
        :param model_cfg_path: Path to model configuration file
        :type model_cfg_path: str
        :param model_file_name: Path to model weights file
        :type model_file_name: str
        :param confidence_threshold: Predictions minimum score
        :type confidence_threshold: str
        :param nms_threshold: Non maximum suppression threshold
        :type nms_threshold: str
        """
        
        with open(model_cfg_path, mode="rb") as f:
            self.cfg = pickle.load(f)
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold
        self.predictor = DefaultPredictor(self.cfg)
        
    def predict(
        self,
        image : ndarray,
        scale_factor: int = 1,
    ) -> Tuple[Dict[str, torch.TensorType], ndarray]: 
        """
        Predicts bounding boxes and masks for given image
        
        :param image: Image to predict
        :type image: ndarray
        :param scale_factor: Scale factor for image to improve performance
        :type scale_factor: int, default 1
        :return: Scaled image and the predictions
        :rtype: Tuple[ndarray, Dict[str, TensorType]]
        """
        
        scale_by = (
            image.shape[1] if image.shape[1] > image.shape[0]
            else image.shape[0]
        )
        image = letterbox(
                image, 
                int(scale_by/scale_factor), 
                stride=64,
                auto=True
        )[0]
        
        torch.cuda.empty_cache()
        
        return image, self.predictor(image)
     