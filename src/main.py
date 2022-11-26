
from utils.config import load_config
from hold_detection.Detector import InstanceSegmentator
from data_operations import labelbox_ops

def main():
    
    global_config, local_config = load_config()
    
    client = labelbox_ops.get_labelbox_client(
        global_config["LABELBOX_API_KEYS"]["hold-detector"]
    )
    labels = labelbox_ops.get_project_labels(client, "hold-detector")
    
    coco_labels = labelbox_ops.labels_to_COCO_format(
        labels, local_config["PATH"]["labelbox_api_routes"]
    )
    
    
    
if __name__ == "__main__":
    main()
