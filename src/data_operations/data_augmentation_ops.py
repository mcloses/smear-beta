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
            A.Transpose(p=P),
            # A.RandomGridShuffle(grid=(2,3), always_apply=True, p=P),
            A.Perspective(p=P),
            A.HorizontalFlip(p=P),
            #A.ElasticTransform(p=P),
            A.RandomCrop(
                height=int(image.shape[0]*0.3),
                width=int(image.shape[1]*0.3),
                p=P
            ),
            A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=10, p=P),
            A.Emboss(p=P),
            A.Downscale(p=P),
            A.ColorJitter(hue=0.5, p=P)
        ],
        bbox_params=A.BboxParams(format='coco', label_fields=["hold_class"], min_area=(-1.0), min_visibility=(-1.0)),
    )
    
    # no_bbox_transform = A.Compose(
    #     [
    #         A.CoarseDropout(max_holes=25, max_height=24, max_width=24, p=P),
    #     ]
    # )
    masks_transform = A.Compose(
        [
            A.ChannelShuffle(p=P),
            A.ColorJitter(hue=1, p=P),
        ]   
    )
    
    transformed_g = general_transform(
        image=image,
        masks=masks,
        bboxes=bboxes,
        hold_class=labels,
    )
    
    # transformed = no_bbox_transform(
    #     image=transformed_g["image"],
    #     masks=transformed_g["masks"],
    #     hold_class=labels,
    # )
    
    return (
        transformed_g["image"],
        transformed_g["masks"],
        transformed_g["bboxes"],
        transformed_g["hold_class"]
    )
    
    