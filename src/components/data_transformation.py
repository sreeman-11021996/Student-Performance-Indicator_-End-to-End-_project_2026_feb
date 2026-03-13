import os, sys

from src.components.data_ingestion import Data_Ingestion
from src.constants import *
from src.utils import get_categorical_columns, get_numerical_columns, get_X_and_y

from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


import pandas as pd
from dataclasses import dataclass


@dataclass
class Data_Transformation_Config:
    transformed_dir : str = os.path.join(ARTIFACT_DIR,DATA_TRANSFORMATION_DIR)
    transformed_train_file_path : str = os.path.join(transformed_dir,TRANSFORMED_TRAIN_DATA_FiILENAME)
    transformed_test_file_path : str = os.path.join(transformed_dir,TRANSFORMED_TEST_DATA_FiILENAME)
    

class Data_Transformation:
    def __init__(self, ingestion_artifact:tuple[str,str]):
        self.train_file_path, self.test_file_path = ingestion_artifact
           
    
    
    
    def initiate_data_transforamtion(self):
        logging.info("Entered the data transformation method or component")
        try:
            
            train_data = pd.read_csv(self.train_file_path)
            test_data = pd.read_csv(self.test_file_path)
            logging.info("Reading the train and test files into Data Frames")
            
            # seperate the dependent features and target column in train and test
            X_train, y_train = get_X_and_y(train_data)
            X_test, y_test = get_X_and_y(test_data)
            logging.info("Seperating the train and test data's into (X_train, y_train) and (X_test and y_test)")
            
            # get categorical & numerical columns
            cat_features = get_categorical_columns(data = X_train)
            num_features = get_numerical_columns(data = X_train)
            
            # initiate the feture transformer objects
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            
            # create a Custom Transformer based on the numeric and categorical transformers
            preprocessor = ColumnTransformer([
                ("OneHotEncoder", oh_transformer, cat_features),
                ("StandardScaler", numeric_transformer, num_features)
                ])
            
            # apply the custom transformer
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.fit_transform(X_test)

            # save the (X_train, y_train) and (X_test and y_test) data
            X_train_transformed.to_csv(self.ingestion_config.train_data_path)
            logging.info(f"Saved train data in dir : {self.ingestion_config.train_data_path}")
            test_set.to_csv(self.ingestion_config.test_data_path)
            logging.info(f"Saved test data in dir : {self.ingestion_config.test_data_path}")
            
            logging.info(f"Inmgestion of the data is completed")
            
            return ()
        
        except Exception as e:
            raise CustomException(e,sys)