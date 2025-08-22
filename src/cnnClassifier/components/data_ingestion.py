import os
import zipfile
import requests
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        '''
        Fetch data from the URL (Azure Blob / direct link)
        '''
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)

            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            response = requests.get(dataset_url, stream=True)
            response.raise_for_status()  # raise error if download fails

            with open(zip_download_dir, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
            return zip_download_dir

        except Exception as e:
            logger.error(f"Error in downloading file: {e}")
            raise e

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory
        Function returns None
        """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            logger.info(f"Extracted zip file {self.config.local_data_file} into {unzip_path}")

        except Exception as e:
            logger.error(f"Error in extracting file: {e}")
            raise e
