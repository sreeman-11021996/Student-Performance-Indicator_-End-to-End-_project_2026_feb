# Read data from sources, split data
import os, sys
from src.logger import logging
from src.exception import CustomException
from src.constants import *

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass


# Data Ingestion Config : input
@dataclass
class Data_Ingestion_Config:
    artifact_dir : str = os.path.join(ARTIFACT_DIR,DATA_INGESTION_DIR)
    raw_data_path : str = os.path.join(artifact_dir, RAW_DATA_FILENAME)
    train_data_path : str = os.path.join(artifact_dir, TRAIN_DATA_FiILENAME)
    test_data_path : str = os.path.join(artifact_dir, TEST_DATA_FILENAME)
    
    
# Data Ingestion Artifact : output
@dataclass
class Data_Ingestion_Artifact:
    train_data_path : str
    test_data_path : str
    

# Data Ingestion Component
class Data_Ingestion:
    def __init__(self):
        self.data_ingestion_config = Data_Ingestion_Config()
    
    def initiate_data_ingestion(self)-> Data_Ingestion_Artifact:
        logging.info("Entered the data ingestion method or component")
        try:
            # read data
            df = pd.read_csv(DATA_FILE_PATH)
            logging.info("Read the dataset as a DataFrame")
            
            # create the directories
            os.makedirs(self.data_ingestion_config.artifact_dir, exist_ok=True)
            logging.info(f"Created artifacts directory: {self.data_ingestion_config.artifact_dir}")
            
            # save raw data
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Saved raw data in dir : {self.data_ingestion_config.raw_data_path}")
              
            # split the data
            train_set, test_set = train_test_split(df, test_size=TEST_SIZE, random_state=42)

            # save the train and test data
            train_set.to_csv(self.data_ingestion_config.train_data_path)
            logging.info(f"Saved train data in dir : {self.data_ingestion_config.train_data_path}")
            test_set.to_csv(self.data_ingestion_config.test_data_path)
            logging.info(f"Saved test data in dir : {self.data_ingestion_config.test_data_path}")
            
            data_ingestion_artifact = Data_Ingestion_Artifact(
                train_data_path=self.data_ingestion_config.train_data_path,
                test_data_path=self.data_ingestion_config.test_data_path
                )
            logging.info(f"Inmgestion of the data is completed")
            
            return data_ingestion_artifact
        
        except Exception as e:
            raise CustomException(e,sys) from None
        