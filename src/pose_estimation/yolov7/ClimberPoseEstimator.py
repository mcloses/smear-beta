import math
import sys

from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch
from torchvision import transforms

from utils.yolo_utils import letterbox
from utils.yolo_utils import non_max_suppression_kpt
from utils.yolo_utils import output_to_keypoint, plot_skeleton_kpts

class ClimberPoseEstimator:
    """
    """
    
    information = {
        "body_parts": {
            "left_hand": {"index": 34, "positions": []},
            "right_hand": {"index": 37, "positions": []},
            "left_foot": {"index": 52, "positions": []},
            "right_foot": {"index": 55, "positions": []},
        },
        "others": {
            "left_knee": {"index": 28},
        }
    }
    
    def __init__(
        self,
        model_path: str,
        yolov7_repository_path: str,
        device: str = "cuda:0",  
    ):
        sys.path.insert(0, yolov7_repository_path)
        self.device = torch.device(device)
        weigths = torch.load(model_path)
        self.model = weigths['model']
        self.model = self.model.half().to(device)
        _ = self.model.eval()
        
    def estimate_pose(
        self,
        frames: List[np.ndarray],
        scale_factor: int = 1,
        save_result: bool = False,
        save_result_path: str = None,
        save_all_skeleton: bool = False,
    ):
        self.frames = frames
        frame_width = int(self.frames[0].shape[1])
        frame_height = int(self.frames[0].shape[0])
        
        scale_by = (
            frame_width if frame_width > frame_height
            else frame_height
        )
        
        if save_result:
            vid_write_image = letterbox(
                self.frames[0],
                int(scale_by/scale_factor), stride=64, auto=True
            )[0]
            resize_height, resize_width = vid_write_image.shape[:2]
            out = cv2.VideoWriter(
                save_result_path,
                cv2.VideoWriter_fourcc(*'mp4v'), 
                30,
                (resize_width, resize_height)
            )
        
        self.information["others"]["foot_distance_treshold"] = 0
        foot_distance_set = False

        for frame in self.frames:
            self.frames.append(frame)
            orig_image = frame
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            image = letterbox(
                image, 
                int(scale_by/scale_factor), stride=64, auto=True
            )[0]
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            image = image.to(self.device)
            image = image.half()
            with torch.no_grad():
                output, _ = self.model(image)    
            output = non_max_suppression_kpt(
                output, 0.25, 0.65, 
                nc=self.model.yaml['nc'], 
                nkpt=self.model.yaml['nkpt'], 
                kpt_label=True
            )
            output = output_to_keypoint(output)
            nimg = image[0].permute(1, 2, 0) * 255
            nimg = nimg.cpu().numpy().astype(np.uint8)
            nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
            
            self.information["body_parts"]["left_hand"]["positions"].append(
                (output[0, self.information["body_parts"]["left_hand"]["index"]], 
                output[0, self.information["body_parts"]["left_hand"]["index"]+1])
            )
            self.information["body_parts"]["right_hand"]["positions"].append(
                (output[0, self.information["body_parts"]["right_hand"]["index"]], 
                output[0, self.information["body_parts"]["right_hand"]["index"]+1])
            )
            self.information["body_parts"]["left_foot"]["positions"].append(
                (output[0, self.information["body_parts"]["left_foot"]["index"]], 
                output[0, self.information["body_parts"]["left_foot"]["index"]+1])
            )
            self.information["body_parts"]["right_foot"]["positions"].append(
                (output[0, self.information["body_parts"]["right_foot"]["index"]], 
                output[0, self.information["body_parts"]["right_foot"]["index"]+1])
            )
            if not foot_distance_set:
                left_foot_confidence = output[
                    0,
                    self.information["body_parts"]["left_foot"]["index"]+2
                ]
                left_knee_confidence = output[
                    0,
                    self.information["body_parts"]["left_knee"]["index"]+2]
                if (
                    left_foot_confidence>0.75 and
                    left_knee_confidence>0.75
                ):
                    # setting foot distance treshold to half of the distance
                    # between the left foot and left knee
                    self.information["others"]["foot_distance_treshold"] = math.sqrt(
                        (
                            output[
                                0,
                                self.information["body_parts"]["left_foot"]["index"]
                            ] - 
                            output[
                                0,
                                self.information["body_parts"]["left_knee"]["index"]
                            ]
                        ) ** 2
                        + 
                        (
                            output[
                                0,
                                self.information["body_parts"]["left_foot"]["index"]+1
                            ] - 
                            output[
                                0,
                                self.information["body_parts"]["left_knee"]["index"]
                            ]+1
                        ) ** 2
                    ) / 2
                    foot_distance_set = True
            
            if save_result:
                if save_all_skeleton:
                    plot_skeleton_kpts(nimg, output[0, 7:].T, 3)
                else:
                    output = np.concatenate((output[0, 34:40],output[0, 52:58]))
                    plot_skeleton_kpts(nimg, output.T, 3)   
                out.write(nimg)
        if save_result:
            out.release()


    def estimate_significant_frames(
        self,
        save_significant_frames_figure: bool = False,
        save_figures_path: str = None,
        sevgol_frame_window_size: int = 5,
        sevgol_filter_poly_order: int = 2,
    ):
        for part in self.information["body_parts"].keys():
            positions = np.array(
                self.information["body_parts"][part]["positions"]
            )   
            diffs = np.diff(positions, axis=0)
            distances = np.sqrt(np.sum(diffs**2, axis=1))
            smoothed_distances = savgol_filter(
                distances,
                sevgol_frame_window_size,
                sevgol_filter_poly_order
            )

            smooth_mean = np.mean(smoothed_distances)
            flat_indices = np.where(
                smoothed_distances < smooth_mean
            )[0]
            groups = np.split(
                flat_indices, 
                np.where(np.diff(flat_indices) != 1)[0]+1
            )
            flat_zones = [(group[0], group[-1]) for group in groups]
            centers = [
                (start_index + end_index) // 2 
                for start_index, end_index in flat_zones
            ]
            self.information["body_parts"][part]["significant_frames"] = centers
            
            if save_significant_frames_figure:
                _ = plt.figure(figsize=(30, 10))
                plt.plot(smoothed_distances)
                plt.scatter(centers, smoothed_distances[centers], color='red')
                plt.xlabel('Time (frames)')
                plt.ylabel('Normalized ' + part + ' distance moved')
                plt.title('Significant Frames for ' + part)
                plt.savefig(save_figures_path + part + '.png')
                
    def get_estimator_output(self):
        return self.information
    