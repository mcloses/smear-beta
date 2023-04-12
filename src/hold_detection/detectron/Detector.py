import os
import pickle

from typing import Dict

from detectron2.engine import DefaultPredictor
from numpy import ndarray
from torch import TensorType


class InstanceSegmentator:
    """
    Instance segmentator class
    """
    def __init__(
        self,
        model_cfg_path : str,
        model_file_name : str,
        confidence_threshold : float,
        nms_threshold : float
    ):
        """
        Instance segmentator class
        
        :param model_cfg_path: Path to model configuration file
        :type model_cfg_path: str
        :param model_file_name: Model file name
        :type model_file_name: str
        :param confidence_threshold: Predictions minimum score
        :type confidence_threshold: str
        :param nms_threshold: Non maximum suppression threshold
        :type nms_threshold: str
        """
        
        with open(model_cfg_path, mode="rb") as f:
            self.cfg = pickle.load(f)
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, model_file_name)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold
        self.predictor = DefaultPredictor(self.cfg)
        
    def predict(self, image : ndarray) -> Dict[str, TensorType]:
        """
        Predicts bounding boxes and masks for given image
        
        :param image: Image to predict
        :type image: ndarray
        :return: Predictions
        :rtype: dict
        """
        
        return self.predictor(image)
     