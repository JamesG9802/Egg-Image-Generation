"""Fetches data from Kaggle and downloads it to a user-specified folder.
"""
import os
import logging
from typing import Final
import requests
import zipfile
from fetch_dataset_get_args import get_args

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%I:%M:%S %p')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


URL: Final[str] = "https://www.kaggle.com/api/v1/datasets/download/abdullahkhanuet22/eggs-images-classification-damaged-or-not" 
"""The URL to the kaggle API endpoint corresponding to the Kaggle dataset.
"""

ZIP_FILE_NAME: Final[str] = "eggs-images-classification-damaged-or-not.zip"

def main():
    args = get_args()
    target_path: os.PathLike = args.target

    logger.info(f"Downloading Egg Image dataset and saving to '{target_path}'.")

    response: requests.Response = requests.get(URL, allow_redirects=True)

    if os.path.exists(target_path) and not os.path.isdir(target_path):
        logger.error(f"Target path '{target_path}' is not a directory.")
        return 1
    if not os.path.exists(target_path):
        os.mkdir(target_path)
        if not os.path.exists(target_path):
            logger.error(f"Target path '{target_path}' doesn't exist and it couldn't be automatically generated.")
            return -1
    
    zip_output_path: os.PathLike = os.path.join(target_path, ZIP_FILE_NAME)

    #   Saving zipped dataset.
    with open(zip_output_path, 'wb') as file:
        file.write(response.content)
    file.close()

    #   Unzipping dataset into contents
    with zipfile.ZipFile(zip_output_path, "r") as zip_ref:
        zip_ref.extractall(target_path)
    zip_ref.close()

    #   Deleting zip
    os.remove(zip_output_path)

    return 0

if __name__ == "__main__":
    exit(main())

