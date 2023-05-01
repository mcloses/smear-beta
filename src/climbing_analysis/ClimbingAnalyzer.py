from typing import Dict, List
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from torch import TensorType

from smear_beta_utils.yolo_utils import letterbox
from smear_beta_utils.detectron_utils import plot_predictions

class ClimbingAnalyzer:
    
    def __init__(
        self,
        video_hold_predictions: Dict[str, TensorType],
        holds_ids: List[int],
    ):
        self.point_to_hold_id_dict = {
            point_tuple : id
            for id in holds_ids
            for point_tuple in list(
                map(
                    tuple,
                    cv2.findContours(
                        (video_hold_predictions["instances"][id]
                            .get("pred_masks")
                            .cpu()
                            .numpy()[0] * 255
                        ).astype(np.uint8),
                        cv2.RETR_TREE,
                        cv2.CHAIN_APPROX_SIMPLE
                    )[0][0].squeeze().tolist()
                )
            )
        }

        hold_polygon_points = np.array(
            list(
                self.point_to_hold_id_dict.keys()
            )
        )
        self.tree = cKDTree(
            hold_polygon_points,
            compact_nodes=False,
            balanced_tree=False
        )
        self.video_hold_predictions = video_hold_predictions
        
    def analyze(
        self,
        pose_estimation_output: dict,
        save_results: bool = False,
        save_path: str = None,
        frames: List[np.ndarray] = None,
        scale_factor: int = 1,
    ):
        foot_threshold = (
            pose_estimation_output["others"]["foot_distance_treshold"]
        )
        analysis = {}

        for part, part_info in pose_estimation_output["body_parts"].items():
            analysis[part] = {}
            
            for sign_frame in part_info["significant_frames"]:
                part_position =  part_info["positions"][sign_frame]
                dist, idx = self.tree.query(part_position, p=2)
                
                if not "foot" in part or dist < foot_threshold:
                    hold_id = self.point_to_hold_id_dict[
                        list(self.point_to_hold_id_dict.keys())[idx]
                    ]
                    if not hold_id in analysis[part]:       
                        analysis[part][hold_id] = sign_frame
                        
                        if save_results:
                            frame_height, frame_width = (
                                frames[sign_frame]
                                .shape[:2]
                            )
                                    
                            scale_by = (
                                int(frame_width) if frame_width > frame_height
                                else int(frame_height)
                            )

                            significant_frame = letterbox(
                                frames[sign_frame].copy(),
                                int(scale_by/scale_factor),
                                stride=64,
                                auto=True
                            )[0]
                            
                            plotted_frame = plot_predictions(
                                significant_frame.copy(),
                                self.video_hold_predictions["instances"][hold_id]
                            )
                            cv2.circle(
                                plotted_frame,
                                (int(part_position[0]),
                                int(part_position[1])),
                                3,
                                (0, 255, 0),
                                -1
                            )

                            title = (
                                str(sign_frame) +
                                " " + part +
                                " " + str(hold_id)
                            )
                            filename = (
                                title + ".jpg"
                            )
                            fig = plt.figure(figsize=(12,7))
                            plt.title(title)
                            plt.axis('off')
                            plt.imshow(plotted_frame)
                            fig.savefig(
                                str(
                                    Path(save_path).joinpath(
                                        filename)
                                ),
                                bbox_inches='tight',
                                dpi=300
                            )
                            