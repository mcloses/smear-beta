
from pathlib import Path
from configparser import ConfigParser
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


config_file = "home.ini"
local_config_path = "smear-beta\\config\\config.ini"
home_path = Path.home()

global_config = ConfigParser()
local_config = ConfigParser()
global_config.read(home_path.joinpath(config_file))
local_config.read(
    Path(global_config["PATH"]["tfg_dir"])
    .joinpath(local_config_path)
)

from hold_detection.Detector import InstanceSegmentator

def main():
    
    image_name = "IMG-20221007-WA0006.jpg"
    detector = InstanceSegmentator()
    
    detector.on_image(
        str(Path(local_config["PATH"]["routes"])
        .joinpath(image_name))
    )
    
if __name__ == "__main__":
    main()