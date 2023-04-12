import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import albumentations as A
from albumentations.core.composition import Compose
from data_operations import coco_labels_ops


class DataAugmentator:
    """
    A class for augmenting image data in COCO format.
    """

    def __init__(
        self,
        coco_labels: Dict,
        num_rounds: int,
        images_path: str,
        images_save_base_id: int,
        labels_save_path: str,
        labels_save_file_name: str,
    ):
        """
        Initializes the DataAugmentator object. Uses the albumentations library
        for data augmentation.

        :param coco_train_labels: A dictionary representing the COCO-formatted
            training labels.
        :type coco_train_labels: dict
        :param num_rounds: The number of rounds of data augmentation to perform.
        :type num_rounds: int
        :param images_path: Path to folder containing image files
        :type images_path: str
        :param images_save_base_id: Base id for augmented images filename, auto-incremented.
                                    Chose one big enough to not conflict with existing images.
        :type images_save_base_id: int
        :param labels_save_path: Path to folder where new COCO labels file will be saved.
        :type labels_save_path: str
        :param labels_save_file_name: Name of new COCO labels file.
        :type labels_save_file_name: str
        """
        
        
        self.coco_labels = coco_labels
        self.num_rounds = num_rounds
        self.images_path = images_path
        self.images_save_base_id = images_save_base_id
        self.labels_save_path = labels_save_path
        self.labels_save_file_name = labels_save_file_name
       
         
    def __get_default_transforms(
        self, 
        image: np.ndarray, 
        transform_probability: float
    ) -> Compose:
        
        return A.Compose(
            [
            A.RandomCrop(
                height=int(image.shape[0]*0.5),
                width=int(image.shape[1]*0.5),
                p=transform_probability
            ),
            A.Transpose(p=transform_probability),
            A.Perspective(p=transform_probability),
            A.HorizontalFlip(p=transform_probability),
            A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=10, p=P),
            A.Emboss(p=transform_probability),
            A.Downscale(p=transform_probability),
            A.ColorJitter(hue=0.5, p=transform_probability)
            ],
            bbox_params=A.BboxParams(
                format='coco', 
                label_fields=["hold_class"], 
                min_area=(-1.0), 
                min_visibility=(-1.0)),
        )
        
    def __apply_transfrom(
        self,
        image: np.array,
        masks: List[np.ndarray],
        bboxes: List[np.ndarray],
        labels: List[str],
        transforms: Compose,
        transform_probability: float
    ) -> Tuple[np.array, List[np.ndarray], List[np.ndarray]]:
        
        if transforms is None:
            transforms = self.__get_default_transforms(image, transform_probability)
        
        transformed = transforms(
            image=image,
            masks=masks,
            bboxes=bboxes,
            hold_class=labels,
        )
    
        return (
            transformed["image"],
            transformed["masks"],
            transformed["bboxes"],
            transformed["hold_class"]
        )

    def create_augmented_samples(
        self,
        transforms : Optional[Compose] = None,
        transform_probability: Optional[float] = 0.2) -> Dict:
        """
        Creates augmented samples from the COCO-formatted training labels.
        
        If transforms is None, the default transforms will be used.
        This are the default transforms:
        

        - albumentations.RandomCrop(), 50% of the original image 
        - albumentations.Transpose()
        - albumentations.Perspective()
        - albumentations.HorizontalFlip()
        - albumentations.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=10)
        - albumentations.Emboss()
        - albumentations.Downscale()
        - albumentations.ColorJitter(hue=0.5)
        - albumentations.BboxParams(
                format='coco', 
                label_fields=["hold_class"], 
                min_area=(-1.0), 
                min_visibility=(-1.0)
            )
        
        :param transforms: Albumentations Compose object containing transforms to apply.
        :type transforms: albumentations.core.composition.Compose
        :param transform_probability: Probability of applying a each individual transform.
        :type transform_probability: float
        
        :return: A dictionary containing the COCO-formatted augmented training labels.
        :rtype: dict
        """
        
        annotations = defaultdict(list, [])
        images = {}
        for annot in self.coco_labels["annotations"]:
            annotations[annot["image_id"]].append(annot)
        for img in self.coco_labels["images"]:
            images[img["id"]] = img
        t_images = images.copy()
        t_annotations = annotations.copy()

        last_annot_id = self.coco_labels["annotations"][-1]["id"] + 1
        last_image_id = self.images_save_base_id

        for i in range(self.num_rounds):
            for key, values in images.items():
                filename = values["file_name"]
                image = cv2.cvtColor(
                    cv2.imread(
                        self.images_path + filename
                    ),
                    cv2.COLOR_BGR2RGB,
                )

                masks = [annot["segmentation"] for annot in annotations[key]]
                bbox = [annot["bbox"] for annot in annotations[key]]
                labels = [annot["category_id"] for annot in annotations[key]]

                numpy_masks = coco_labels_ops.__coco_polygons_to_numpy_masks(
                    masks, image.shape[:2]
                )

                t_image, t_masks, t_bbox, t_labels = self.__apply_transfrom(
                    image, numpy_masks, bbox,
                    labels, transforms, transform_probability
                )

                t_masks = coco_labels_ops.__numpy_masks_to_coco_polygons(t_masks)

                annotations_copy = []

                for j in range(len(t_masks)):
                    annot_copy = {}
                    annot_copy["area"] = 0
                    annot_copy["bbox"] = list(t_bbox[j])
                    annot_copy["category_id"] = t_labels[j]
                    annot_copy["id"] = last_annot_id
                    annot_copy["image_id"] =  last_image_id
                    annot_copy["iscrowd"] = 0
                    annot_copy["segmentation"] = t_masks[j]

                    annotations_copy.append(annot_copy)
                    last_annot_id += 1
            
                img_copy = values.copy()
                img_copy["id"] = last_image_id
                img_copy["file_name"] = str(last_image_id)+".jpg"
                img_copy["width"] = t_image.shape[1]
                img_copy["height"] = t_image.shape[0]
                
                cv2.imwrite(
                    self.images_path + img_copy["file_name"],
                    t_image, 
                )

                t_annotations[str(last_image_id)] = annotations_copy
                t_images[str(last_image_id)] = img_copy
                
                last_image_id += 1
        
        coco_labels_transformed = self.coco_labels.copy()
        coco_labels_transformed["annotations"] = [subitem for item in list(t_annotations.values()) for subitem in item]
        coco_labels_transformed["images"] = [item for item in list(t_images.values())]
        
        self.coco_labels_transformed = coco_labels_transformed
        
        with open(self.labels_save_path + self.labels_save_file_name, mode="w", encoding="utf-8") as f:
            json.dump(self.coco_labels_transformed, f, ensure_ascii=False, indent=4)
        
        return self.coco_labels_transformed
