"""
Script to extract tagged images from LabelBox through its API,
convert them to COCO format and save them on the data subdirectory
"""


import json
from pathlib import Path

from smear_beta_utils.config import load_config
from data_operations import labelbox_ops

def main():
    
    global_config, local_config = load_config()
    
    client = labelbox_ops.get_labelbox_client(
        API_KEY = global_config["LABELBOX_API_KEYS"]["hold-detector"],
    )
    project = labelbox_ops.get_project(
        client = client,
        project_id = local_config["LABELBOX_PROJECT_IDS"]["hold-detector"],
    )
    labels = labelbox_ops.get_project_labels(
        project = project,
    )
    coco_labels = labelbox_ops.labels_to_COCO_format(
        labels = labels,
        image_export_path = local_config["PATH"]["labelbox_api_routes"],
    )
    
    # Converting image_root from PosixPath to str to serialize JSON
    coco_labels["info"]["image_root"] = str(coco_labels["info"]["image_root"])
    
    with open(
        file = Path(local_config["PATH"]["coco_labels"]).joinpath("coco_route_labels.json"),
        mode = "w",
        encoding = "utf-8",
    ) as f:
        json.dump(coco_labels, f, ensure_ascii=False, indent=4)    
    
    
if __name__ == "__main__":
    main()
