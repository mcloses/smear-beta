"""
All functions needed to operate with labelbox
data and labels
"""


from labelbox import Client
from labelbox.data.annotation_types.collection import (
    LabelCollection,
    LabelGenerator,
)
from labelbox.data.serialization import COCOConverter
from labelbox.data.serialization.coco.instance_dataset import (
    CocoInstanceDataset,
)


def get_labelbox_client(API_KEY: str) -> Client:
    """
    Initialize Labelbox API Client
    
    :param API_KEY: API key for client
    :type API_KEY: str
    
    :return: A Labelbox API Client
    :rtype: labelbox.Client
    """
    
    return Client(API_KEY)


def get_project_labels(
    client: Client,
    project_name: str
) -> LabelGenerator:
    """
    Get project labels from API
    
    :param client: Labelbox API Client
    :type client: labelbox.Client
    :param project_name: Project name in Labelbox
    :type project_name: str
    
    :return: The project labels
    :rtype: LabelGenerator
    """
    
    project = client.get_project(project_name)
    labels = project.label_generator()
    
    return labels


def labels_to_COCO_format(
    labels: LabelCollection,
    image_export_path: str,
) -> CocoInstanceDataset:
    """
    Convert a Labelbox LabelCollection into an mscoco dataset.
    This function will only convert masks, polygons, and rectangles.
    Masks will be converted into individual instances.
    
    :param labels: Labels object in Labelbox format
    :type labels: labelbox.Collection
    :param image_export_path: Path to export images
    :type image_export_path: str
    
    :return: A dictionary containing labels in the COCO object format
    :rtype: CocoInstanceDataset
    """
    
    coco_labels = COCOConverter.deserialize_instances(
        labels=labels,
        image_root=image_export_path,
    )
    
    return coco_labels
