import os
from typing import Any

from src.logger import logging
from src.exception import CustomException

from src.constants import *
from src.utils import load_object, save_numpy_array_data

import pandas as pd
import numpy as np
from dataclasses import dataclass



@dataclass
class Prediction_Config:  
    artifact_dir : str = os.path.join(ARTIFACT_DIR, PREDICTION_TIME_STAMP)  
    preprocessed_obj_file_path : str = os.path.join(artifact_dir, DATA_TRANSFORMATION_DIR, 
                                                    PREPROCESSED_OBJ_FILENAME)
    model_file_path : str = os.path.join(artifact_dir, MODEL_TRAINER_DIR, MODEL_FILENAME)
    
    y_pred_file_path : str = os.path.join(PREDICTION_DIR, CURRENT_TIME_STAMP, PREDICTION_FILE_NAME)

    

    
@dataclass
class Prediction_Artifact:
    y_pred_file_path : str
    

class Prediction:
    
    def __init__(self, data : pd.DataFrame):
        self.prediction_config = Prediction_Config()
        self.data = data

        
    def load_pipeline_components(self)-> tuple[Any, Any]:
        """
        Function : Load model + preprocessor for inference.
        Return : (model, preprocessor)
        """
        # get file paths
        model_file_path = self.prediction_config.model_file_path
        preprocessor_file_path = self.prediction_config.preprocessed_obj_file_path
        
        # load objects
        model = load_object(file_path=model_file_path)
        preprocessor = load_object(file_path=preprocessor_file_path)
        
        return model, preprocessor
   
    
    def save_prediction(self, y_pred:np.ndarray)-> str:
        # make dir
        y_pred_file_path = self.prediction_config.y_pred_file_path
        y_pred_dir = os.path.dirname(y_pred_file_path)
        os.makedirs(y_pred_dir, exist_ok=True)
        
        # save the file
        save_numpy_array_data(np_array=y_pred, file_path=y_pred_file_path)
        return y_pred_file_path
        
    
    
    def initiate_perdiction(self)-> Prediction_Artifact:
        try:
            logging.info(f"Entered the Prediction method or component")
            
            # preprocessor.pkl & model.pkl
            logging.info(f"Begin loading the preprocessing object & model")
            model, preprocessor = self.load_pipeline_components()
            logging.info(f"Completed loading the preprocessing object & model")
            
            
            # prediction
            logging.info(f"Begin Prediction")
            transformed_data = preprocessor.transform(self.data)
            y_prediction = model.predict(transformed_data)
            logging.info(f"Completed Prediction")
            
            
            # save the prediction
            y_pred_file_path = self.save_prediction(y_pred=y_prediction)
            logging.info(f"Saved the Prediction in : {y_pred_file_path}")
            
            
            prediction_artifact = Prediction_Artifact(y_pred_file_path=y_pred_file_path)
            return prediction_artifact
  
        except Exception as e:
            raise CustomException(e) from None        
