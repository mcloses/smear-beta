
import numpy as np
import cv2
from typing import List, Tuple


def coco_polygons_to_numpy_masks(
    polygons_list: List[List[List[float]]],
    image_size: Tuple[int, int],
    fill_value: int = 255,
    background_value: int = 0,
) -> List[np.ndarray]:
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

def numpy_masks_to_coco_polygons(
    masks_list: List[np.ndarray],
) -> List[List[List[float]]]:
    
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