"""
All functions needed to operate with labelbox
data and labels
"""


from labelbox import Client, Project
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
    
    return Client(api_key=API_KEY)


def get_project(client: Client, project_id: str) -> Project:
    """
    Get project instance from Labelbox
    
    :param client: A Labelbox API Client instance
    :type client: Labelbox.Client
    :param project_id: ID for the Labelbox project to retrieve
    :param project_id: str
    """
    
    return client.get_project(project_id)


def get_project_labels(
    project: Project
) -> LabelGenerator:
    """
    Get project labels from API
    
    :param project: Labelbox API Project
    :type project: labelbox.Project
    
    :return: The project labels
    :rtype: LabelGenerator
    """
        
    return project.label_generator()


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
    
    coco_labels = COCOConverter.serialize_instances(
        labels=labels,
        image_root=image_export_path,
    )
    
    return coco_labels
