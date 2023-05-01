"""
This module provides functions for converting COCO-formatted labels
to uniclasslabels and for converting COCO-style segmentation
annotations to binary masks and vice versa.

The coco_labels_to_uniclass function takes COCO-formatted training
and test labels as inputs and converts the category IDs to uniclass
format. The new labels are saved as two separate JSON files to the
specified folder path. This function returns a tuple containing the
new COCO-formatted training and test labels.

The __coco_polygons_to_numpy_masks function takes a list of segmentation
polygons in COCO format and converts them to binary masks using OpenCV's
cv2.fillPoly function. The resulting masks are returned as a list of
NumPy arrays.

The __numpy_masks_to_coco_polygons function takes a list of binary masks
and converts them to segmentation polygons in COCO format using OpenCV's
cv2.findContours function. The resulting polygons are returned as a list
of lists of lists of floats.
"""

import numpy as np
import cv2
from typing import List, Tuple
import json

def coco_labels_to_uniclass(
    coco_train_labels: dict,
    coco_test_labels: dict,
    labels_save_path: str,
    train_labels_save_file_name: str,
    test_labels_save_file_name: str,
) -> Tuple[dict, dict]:
    """
    Converts COCO-formatted labels to uniclass labels.
    Saves the new labels to the specified path in JSON COCO format.
    
    :param coco_train_labels: A dictionary representing the COCO-formatted
                              training labels.
    :type coco_train_labels: dict
    :param coco_test_labels: A dictionary representing the COCO-formatted
                                test labels.
    :type coco_test_labels: dict
    :param labels_save_path: Path to folder where new COCO labels file will be saved.
    :type labels_save_path: str
    :param train_labels_save_file_name: Name of new COCO training labels file.
    :type train_labels_save_file_name: str
    :param test_labels_save_file_name: Name of new COCO test labels file.
    :type test_labels_save_file_name: str
    
    :return: A tuple containing the new COCO-formatted training and test labels.
    :rtype: Tuple[dict, dict]
    """

    for dataset in [coco_train_labels, coco_test_labels]:
        annotations = []
        for annotation in dataset["annotations"]:
            annotation["category_id"] = 1
            annotations.append(annotation)
        dataset["annotations"] = annotations
        dataset["categories"] = [dataset["categories"][0]]
        
    with open(labels_save_path+train_labels_save_file_name, mode="w", encoding="utf-8") as f:
        json.dump(coco_train_labels, f, ensure_ascii=False, indent=4)
    with open(labels_save_path+test_labels_save_file_name, mode="w", encoding="utf-8") as f:
        json.dump(coco_test_labels, f, ensure_ascii=False, indent=4)
    
    return coco_train_labels, coco_test_labels

def __coco_polygons_to_numpy_masks(
    polygons_list: List[List[List[float]]],
    image_size: Tuple[int, int],
    fill_value: int = 255,
    background_value: int = 0,
) -> List[np.ndarray]:
    """
    Convert COCO-style segmentation annotations to binary masks.

    :param polygons_list: A list of segmentation polygons.
    :type polygons_list: List[List[List[float]]]
    :param image_size: The size of the image that the segmentation annotations refer to.
    :type image_size: Tuple[int, int]
    :param fill_value: The value to fill the inside of the polygons with.
    :type fill_value: int
    :param background_value: The value to fill the outside of the polygons with.
    :type background_value: int
    :return: A list of binary masks representing the segmentation annotations.
    :rtype: List[np.ndarray]
    """
    poly_masks = [
        np.reshape(mask, (-1, 2)) for mask in polygons_list 
    ]
    filled_masks = []
    for mask in poly_masks:
        if not background_value:
            array = np.zeros(image_size)
        else:
            array = np.full(image_size, background_value)
        mask = np.array([list(x) for x in list(mask)], np.int32)
        cv2.fillPoly(array, [mask], fill_value)
        filled_masks.append(array)
    
    return filled_masks

def __numpy_masks_to_coco_polygons(
    masks_list: List[np.ndarray]
) -> List[List[List[float]]]:
    """
    Convert binary masks to COCO-style segmentation annotations.

    :param masks_list: A list of binary masks representing segmentation annotations.
    :type masks_list: List[np.ndarray]
    :return: A list of segmentation polygons.
    :rtype: List[List[List[float]]]
    """
    return [
        [
            list(
                cv2.findContours(
                    np.array(mask, dtype=np.uint8),
                    mode=cv2.RETR_TREE,
                    method=cv2.CHAIN_APPROX_NONE
                )[0][0].flatten().astype(float)
            )
        ]
        for mask in masks_list
        if mask.sum()!=0
    ]