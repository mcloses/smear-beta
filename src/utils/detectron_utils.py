"""
Utils functions related to the detectron2 framework and the instance segmentation tasks.
"""

from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

from torch import TensorType

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

def plot_predictions(
    image : np.ndarray,
    predictions : Dict[str, TensorType]
) -> np.ndarray:
    """
    Plot predictions on an input image.

    :param image: An input image in the format of an ndarray with shape (H, W, C).
    :type image: np.ndarray
    :param predictions: A dictionary containing the predicted instances.
    :type predictions: Dict[str, TensorType]
    :return: An ndarray containing the plotted predictions on the input image.
    :rtype: np.ndarray
    """
        
    v = Visualizer(
        image[:,:,::-1],
        metadata={},
        instance_mode=ColorMode.SEGMENTATION,
    )
    v = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    return v.get_image()


def plot_samples(
    dataset_name: str,
    n_images: int = 1,
    image_size: Tuple[int,int] = (20,15),
):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)
    
    for s in random.sample(dataset_custom, n_images):
        img = cv2.imread(s["file_name"])
        v = Visualizer(
            img[:,:,::-1],
            metadata = dataset_custom_metadata,
            scale = 0.5,
        )
        v = v.draw_dataset_dict(s)
        plt.figure(figsize = image_size)
        plt.imshow(v.get_image())
        plt.show()

      
def get_train_cfg(
    config_file_path: str,
    checkpoint_url: str,
    train_dataset_name: str,
    test_dataset_name: str,
    n_classes: int,
    device: str,
    output_dir: str,
):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 500
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    
    return cfg
