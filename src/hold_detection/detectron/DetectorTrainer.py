import pickle

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import torch

class DetectorTrainer:
    """
    Class for training an object detector model using Detectron2.
    """
    
    def __init__(
        self, 
        config_file_save_path: str,
        model_file_save_path: str,
        n_classes: int,
        train_images_path: str,
        train_labels_path: str,
        test_images_path: str,
        test_labels_path: str,
        device: str = "cuda",
        pre_trained_model_url: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        train_dataset_name: str = "train_dataset",
        test_dataset_name: str = "test_dataset"
    ):
        """
        Class for training an object detector model using Detectron2.
        
        :param config_file_save_path: The path to save the model configuration file.
        :type config_file_save_path: str
        :param model_file_save_path: The path to save the trained model checkpoint.
        :type model_file_save_path: str
        :param n_classes: The number of classes in the training data.
        :type n_classes: int
        :param train_images_path: The path to the directory containing the training images.
        :type train_images_path: str
        :param train_labels_path: The path to the file containing the training labels in COCO format.
        :type train_labels_path: str
        :param test_images_path: The path to the directory containing the test images.
        :type test_images_path: str
        :param test_labels_path: The path to the file containing the test labels in COCO format.
        :type test_labels_path: str
        :param device: The device to use for training. Defaults to "cuda".
        :type device: str
        :param pre_trained_model_url: The URL to the pre-trained model checkpoint.
        :type pre_trained_model_url: str
        :param train_dataset_name: The name of the training dataset.
        :type train_dataset_name: str
        :param test_dataset_name: The name of the test dataset.
        :type test_dataset_name: str
        """
    
        self.checkpoint_url = pre_trained_model_url
        self.cfg_save_path = config_file_save_path
        self.output_dir = model_file_save_path
        self.num_classes = n_classes
        self.device = device
        self.train_dataset_name = train_dataset_name
        self.train_images_path = train_images_path
        self.train_labels_path = train_labels_path
        self.test_dataset_name = test_dataset_name
        self.test_images_path = test_images_path
        self.test_labels_path = test_labels_path
        
    def __get_train_cfg(
        self,
        learning_rate,
        max_iter,
        num_workers,
        ims_per_batch
    ):
        
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(self.checkpoint_url))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.checkpoint_url)
        self.cfg.DATASETS.TRAIN = (self.train_dataset_name ,)
        self.cfg.DATASETS.TEST = (self.test_dataset_name,)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.n_classes
        self.cfg.MODEL.DEVICE = self.device
        self.cfg.OUTPUT_DIR = self.output_dir
        self.cfg.SOLVER.BASE_LR = learning_rate
        self.cfg.SOLVER.MAX_ITER = max_iter
        self.cfg.DATALOADER.NUM_WORKERS = num_workers
        self.cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
        self.cfg.SOLVER.STEPS = []

    def train(
        self,
        learning_rate : float = 0.001,
        max_iter : int = 500,
        num_workers : int = 2,
        ims_per_batch : int = 2
    ):
        """
        Train the object detector model.
        
        :param learning_rate: The learning rate for the training. Defaults to 0.001.
        :type learning_rate: float
        :param max_iter: The maximum number of iterations for the training. Defaults to 500.
        :type max_iter: int
        :param num_workers: The number of workers to use for data loading. Defaults to 2.
        :type num_workers: int
        :param ims_per_batch: The number of images per batch for the training. Defaults to 2.
        :type ims_per_batch: int
        """
        
        register_coco_instances(
            name=self.train_dataset_name,
            metadata={},
            json_file=self.train_annotations_path,
            image_root=self.train_images_path,
        )
        register_coco_instances(
            name=self.test_dataset_name,
            metadata={},
            json_file=self.test_annotations_path,
            image_root=self.test_images_path,
        )

        self.__get_train_cfg(
            learning_rate, max_iter,
            num_workers, ims_per_batch
        )

        with open(self.cfg_save_path, "wb") as f:
            pickle.dump(self.cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

        torch.cuda.empty_cache()
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()