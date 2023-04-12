"""
Functions relating to data augmentation
"""

import numpy
import albumentations as A
from typing import List, Tuple

P = 0.2

def apply_transform(
    image: numpy.array,
    masks: List[numpy.ndarray],
    bboxes: List[numpy.ndarray],
    labels: List[str],
)-> Tuple[numpy.array, List[numpy.ndarray], List[numpy.ndarray]]:
    
    general_transform = A.Compose(
        [
            A.RandomCrop(
                height=int(image.shape[0]*0.5),
                width=int(image.shape[1]*0.5),
                p=P
            ),
            A.Transpose(p=P),
            A.Perspective(p=P),
            A.HorizontalFlip(p=P),
            A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=10, p=P),
            A.Emboss(p=P),
            A.Downscale(p=P),
            A.ColorJitter(hue=0.5, p=P)
        ],
        bbox_params=A.BboxParams(format='coco', label_fields=["hold_class"], min_area=(-1.0), min_visibility=(-1.0)),
    )
    
    transformed_g = general_transform(
        image=image,
        masks=masks,
        bboxes=bboxes,
        hold_class=labels,
    )
    
    return (
        transformed_g["image"],
        transformed_g["masks"],
        transformed_g["bboxes"],
        transformed_g["hold_class"]
    )
    
    