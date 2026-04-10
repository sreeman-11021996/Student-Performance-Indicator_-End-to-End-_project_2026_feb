import os, sys

from src.components.data_ingestion import Data_Ingestion_Artifact
from src.constants import *
from src.utils import save_object, save_numpy_array_data

from src.exception import CustomException
from src.logger import logging

# data transformation tools
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


import pandas as pd
import numpy as np
from dataclasses import dataclass


# Data Transformation Config : input (where to save files)
@dataclass
class Data_Transformation_Config:
    # artifact/time_stamp/data_transformation
    artifact_dir : str = os.path.join(ARTIFACT_DIR, CURRENT_TIME_STAMP, DATA_TRANSFORMATION_DIR)
    preprocessed_obj_file_path : str = os.path.join(artifact_dir,PREPROCESSED_OBJ_FILENAME)
    transformed_train_file_path : str = os.path.join(artifact_dir,TRANSFORMED_TRAIN_DATA_FiILENAME)
    transformed_test_file_path : str = os.path.join(artifact_dir,TRANSFORMED_TEST_DATA_FiILENAME)
    # .....
    
    
# Data Transformation Artifact : output (save files)
@dataclass
class Data_Transformation_Artifact:
    preprocessed_obj_file_path : str
    train_arr_file_path : str
    test_arr_file_path : str


class Data_Transformation:
    def __init__(self, data_ingestion_artifact:Data_Ingestion_Artifact):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_config = Data_Transformation_Config() 
    
    # Check imbalance in data
    # def check_data_imbalance () -> checks for imbalance in data & returns balanced data   
    
    
    @staticmethod
    def get_categorical_columns (data:pd.DataFrame)->list[str]:
        """Return a list of column names that are categorical."""
        try:
            cat_cols = data.select_dtypes(include='object').columns
            return cat_cols
        
        except Exception as e:
            raise CustomException(e) from None


    @staticmethod
    def get_numerical_columns (data:pd.DataFrame)->list[str]:
        """Return a list of column names that are numerical."""
        try:
            num_cols = data.select_dtypes(exclude='object').columns
            return num_cols
        
        except Exception as e:
            raise CustomException(e) from None
    
    
    @staticmethod     
    def separate_features_and_target (data:pd.DataFrame, target_column:str)->tuple[pd.DataFrame, pd.Series]:
        """
        Args:
            data (pd.DataFrame): complete dataframe
            target_column (str):

        Returns:
            X : Feature Dataframe
            y : target Series
        """
        try:
            X = data.drop(columns=[target_column],axis=1)
            Y = data[target_column]
            return X, Y
        
        except Exception as e:
            raise CustomException(e) from None

    
    def get_data_transformer_object(self, data:pd.DataFrame)->ColumnTransformer:
        """Gives me the data transformation object to apply on data

        Args:
            data (pd.DataFrame): X_train

        Returns:
            ColumnTransformer: preprocessor
            
        Transformation:
            Missing Values
            Scaling data
            categorical - One hot encoding
        """
        try:
            # get categorical & numerical columns
            cat_features = self.get_categorical_columns(data = data)
            num_features = self.get_numerical_columns(data = data)
            
            # create categorical and numerical pipelines
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)) # (x/std) to keep the sparcity intact
                ]
            )
            logging.info(f"Categorical Pipeline Constructed : {cat_features}")
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )           
            logging.info(f"Numerical Pipeline Constructed : {num_features}")
            
            
            # create the preprocessor object
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline, num_features),
                    ("categorical_pipeline", cat_pipeline, cat_features)
                ]
            )
            logging.info("Preprocessor object created.")
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e) from None
        
    
    def initiate_data_transforamtion(self)->Data_Transformation_Artifact:       
        logging.info("Entered the data transformation method or component")
        try:            
            train_df = pd.read_csv(self.data_ingestion_artifact.train_data_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_data_path)
            logging.info("Reading the train and test files into Data Frames")
            
            # seperate the dependent features and target column in train and test
            input_feature_train_df, target_feature_train_df = self.separate_features_and_target(
                                                                data=train_df, target_column=TARGET_COLUMN)
            input_feature_test_df, target_feature_test_df = self.separate_features_and_target(
                                                                data=test_df, target_column=TARGET_COLUMN)
            logging.info("Seperating the train and test data's into (input_feature_train_df, target_feature_train_df) and "
                "(input_feature_test_df, target_feature_test_df)")
            
            
            # get preprocesser object 
            preprocessing_obj = self.get_data_transformer_object(data = input_feature_train_df)
            
            # apply the custom transformer
            input_feature_transformed_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_transformed_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")



            # concatinate the arrays of X and y into one array 
            train_arr = np.c_[input_feature_transformed_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_transformed_test_arr, np.array(target_feature_test_df)]
            
            
            # get the file paths to train_arr, test_arr, preprocessed_obj
            os.makedirs(self.data_transformation_config.artifact_dir, exist_ok=True)
            train_arr_file_path = self.data_transformation_config.transformed_train_file_path
            test_arr_file_path = self.data_transformation_config.transformed_test_file_path
            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_obj_file_path
            
            
            # save the train and test arrays
            save_numpy_array_data(np_array=train_arr, file_path= train_arr_file_path)
            logging.info(f"Saved train numpy array in dir : {train_arr_file_path}")
            save_numpy_array_data(np_array=test_arr, file_path= test_arr_file_path)
            logging.info(f"Saved train numpy array in dir : {test_arr_file_path}")
            
            # save the preprocessing object
            save_object(object=preprocessing_obj, file_path=preprocessing_obj_file_path)
            logging.info(f"Saved preprocessing object in dir : {preprocessing_obj_file_path}")
            
            data_transformation_artifact = Data_Transformation_Artifact(
                preprocessed_obj_file_path = preprocessing_obj_file_path,
                train_arr_file_path = train_arr_file_path,
                test_arr_file_path = test_arr_file_path 
            )
            logging.info(f"Transformation of the data is completed")
            
            return data_transformation_artifact
        
        except Exception as e:
            raise CustomException(e) from None