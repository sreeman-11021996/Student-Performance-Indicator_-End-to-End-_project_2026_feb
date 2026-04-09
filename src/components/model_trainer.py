# general
import os
from typing import Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict

# own methods
from src.logger import logging
from src.exception import CustomException
from src.constants import *
from src.utils import save_object, get_best_model_name, get_train_and_test_arrays, evaluate_models
from src.components.data_transformation import Data_Transformation_Artifact


# algorithms
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# data operation
import numpy as np



@dataclass
class Model_Trainer_Config:
    artifact_dir : str = os.path.join(ARTIFACT_DIR, CURRENT_TIME_STAMP, MODEL_TRAINER_DIR)
    trained_model_file_path : str = os.path.join(artifact_dir, TRAINED_MODEL_FILENAME)
    

@dataclass
class Model_Trainer_Artifact:
    """
    trained_model_metrics = {'cval_r2_score' : 0.0, 'overfit_gap' : 0.0, 'cval_r2_score_std' : 0.0, 
    'test_r2_score' : 0.0}
    field(default_factory= lambda : defaultdict(float)) == gives me a new dictionary with float values 
    every new instance.
    """
    trained_model_name : str
    trained_model_file_path : str
    trained_model_metrics : Dict[str, float] = field(default_factory= lambda : defaultdict(float))

 
    
class Model_Trainer:
    def __init__(self, data_transformation_artifact:Data_Transformation_Artifact):
        self.data_transformation_arifact = data_transformation_artifact
        self.model_trainer_config = Model_Trainer_Config()
      
                
    def create_model_trainer_artifact(self, best_model:Any, trained_models:dict, best_model_name:str, 
                                      x_train:np.ndarray, y_train:np.ndarray, 
                                      x_test:np.ndarray, y_test: np.ndarray)-> Model_Trainer_Artifact:
        """  
        Arguments :
            best_model : Any sklearn-compatible regressor (LinearRegression, XGBoost, CatBoost, etc.) 
            with .fit/.predict/.score()
        
        Returns:
            Model_Trainer_Artifact: 
                trained_model_metrics = {'r2_score' : 0.0, 'overfit_gap' : 0.0, 
                'cval_r2_score_std' : 0.0, 'test_r2_score' : 0.0}
        """
        try:
            
            best_model.fit(x_train, y_train)
            
            test_r2_score = best_model.score(x_test, y_test)
            
            best_model_metrics = {
                CVAL_R2_SCORE : trained_models[CVAL_R2_SCORE][best_model_name],
                OVERFIT_GAP : trained_models[OVERFIT_GAP][best_model_name],
                CVAL_R2_SCORE_STD : trained_models[CVAL_R2_SCORE_STD][best_model_name],
                TEST_R2_SCORE : test_r2_score
            }
            
            # save model in model file path
            best_model_file_path = self.model_trainer_config.trained_model_file_path
            save_object(object = best_model, file_path = best_model_file_path)
            logging.info(f"Saved the model in {best_model_file_path}")
            
            # create model trainer artifact
            model_trainer_artifact = Model_Trainer_Artifact(
                trained_model_name = best_model_name,
                trained_model_file_path = best_model_file_path,
                trained_model_metrics = best_model_metrics
            )
            logging.info("Created the Model Trainer Artifact") 
            return model_trainer_artifact
        
        except Exception as e:
            raise CustomException(e) from None
               
        
    def initiate_model_trainer(self)-> Model_Trainer_Artifact:
        try:
            logging.info(f"Initaited the model trainer component")
            
            # get the train and test
            logging.info(f"start the loading of train and test arrays from artifacts dir")
            train_file_path = self.data_transformation_arifact.train_arr_file_path
            test_file_path = self.data_transformation_arifact.test_arr_file_path 
            
            (x_train, y_train), (x_test, y_test) = get_train_and_test_arrays(
                                                    train_file_path=train_file_path, 
                                                    test_file_path=test_file_path)
            
            # models dict 
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression" : Ridge(),
                "Lasso Regression" : Lasso(),
                "k-Neighbours Regression" : KNeighborsRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False)
            }
            
            # evaluate the models dict
            logging.info(f"Start evaluation of all the models")
            trained_models_report = evaluate_models(x_train=x_train, y_train=y_train, models=models)
            
            # get best model name
            logging.info("Start the best model comparision")
            best_model_name = get_best_model_name(trained_models=trained_models_report)
            
            # create model trainer artifact
            logging.info(f"Start the model trainer artifact creation")
            best_model = models[best_model_name]
            model_trainer_artifact = self.create_model_trainer_artifact(
                                        best_model= best_model,trained_models=trained_models_report,
                                        best_model_name=best_model_name, x_train=x_train, y_train=y_train, 
                                        x_test=x_test, y_test=y_test)
            
            return model_trainer_artifact
        
        except Exception as e:
            raise CustomException(e) from None