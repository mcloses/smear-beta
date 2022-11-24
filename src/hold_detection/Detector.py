"""
Instance segmentor class
"""

import numpy as np
import cv2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo



class InstanceSegmentator:
    def __init__(self):
        
        self.cfg = get_cfg()
        
        #Load model config and pretrained model:
        
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
            )
        )
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        )
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(self.cfg)


    def on_image(self, imagePath: str):
        
        image = cv2.imread(imagePath)
        cv2.imshow("Image", image)
        cv2.waitKey()
        
        predictions = self.predictor(image)
        viz = Visualizer(
            image[:,:,::-1],
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
            instance_mode = ColorMode.IMAGE_BW,
        )
        output = viz.draw_instance_predictions(
            predictions["instances"].to("cpu")
        )
        cv2.imshow("Result", output.get_image()[:,:,::-1])
        cv2.waitKey()
        