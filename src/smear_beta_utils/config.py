"""
Global and local configuration functions
"""

import sys

from typing import Tuple

from pathlib import Path
from configparser import ConfigParser


def load_config(
    project_name: str = "smear-beta",
    global_config_file_path: Path = Path.home(),
    global_config_filename: str = "home.ini",
    local_config_subdirectory: Path = Path("config\\"),
    local_config_filename: str = "config.ini",
) -> Tuple[ConfigParser, ConfigParser]:
    """
    Load global configuration dict and local project
    configuration dict
    
    :param global_config_file_path: Path to global ini config file,
                                    defaults to global variable $HOME
    :type global_config_file_path: pathlib.Path
    :param global_config_filename: Global config filename, defaults to "home.ini"
    :type global_config_filename: str
    :param local_config_subdirectory: Path to local config file within
                                      project, defaults to "smear-beta\\config\\"
    :type local_config_subdirectory: pathlib.Path
    :param local_config_filename: Local config filename, defaults to "config.ini"
    
    :return: Tuple of global and local configurations parsed into ConfigParser dicts
    :rtype: Tuple[ConfigParser, ConfigParser]
    """
    
    global_config, local_config = ConfigParser(), ConfigParser()
    global_config.read(
        global_config_file_path.joinpath(global_config_filename)
    )
    local_config.read(
        Path(global_config["PATH"][project_name])
        .joinpath(local_config_subdirectory)
        .joinpath(local_config_filename)
    )
    
    for path in local_config["PATH"]:
        local_config["PATH"][path] = str(Path(
            global_config["PATH"][project_name]
        ).joinpath(local_config["PATH"][path]))
        
    return global_config, local_config
