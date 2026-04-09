# general
import os
from typing import Dict
from dataclasses import dataclass, field
from collections import defaultdict

# own methods
from src.logger import logging
from src.exception import CustomException
from src.constants import *
from src.utils import save_object, load_numpy_array_data
from src.components.data_transformation import Data_Transformation_Artifact


# metrics
from sklearn.metrics import r2_score, mean_squared_error


# algorithms
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# training
from sklearn.model_selection import KFold, cross_validate

# data operation
import numpy as np



@dataclass
class Model_Trainer_Config:
    artifact_dir : str = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR)
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
      
        
    def get_train_and_test_arrays(self)->tuple:
        """
        Returns:
            tuple: (x_train, y_train), (x_test, y_test) : all are np.ndarrays
        """
        try:
            train_file_path = self.data_transformation_arifact.train_arr_file_path
            test_file_path = self.data_transformation_arifact.test_arr_file_path
            
            train_arr = load_numpy_array_data(file_path = train_file_path)
            test_arr = load_numpy_array_data(file_path = test_file_path)
            logging.info(f"Loaded the train & test arrays from artifacts")
            
            x_train, y_train, x_test, y_test = (train_arr[:,:-1], train_arr[:,-1], 
                                                test_arr[:,:-1], test_arr[:,-1])
            
            return (x_train, y_train), (x_test, y_test)    
    
        except Exception as e:
            raise CustomException(e) from None
        
    
    def evaluate_models(self, x_train:np.ndarray, y_train:np.ndarray, models:dict)-> Dict:
        """
        Arguments : models = {'linear_reg' : LinearRegression(), ...}
        
        Return : trained_models_report = {
            'models' : {'linear_reg' : linear model, ...},
            'cval_r2_score' : {'linear_reg' : .., ...},
            'overfit_gap' : {...},
            'cval_r2_score_std' : ...
            }
        """
        try:
            # set up the report dictionary
            trained_models_report_fields = [MODELS, CVAL_R2_SCORE, OVERFIT_GAP, CVAL_R2_SCORE_STD]
            trained_models_report = dict.fromkeys(trained_models_report_fields, {})
            
            logging.info(f"Starting the kfold cross validation on models : {models.keys()}")
            # loop all the models
            for model_name, model in models.items():
                # kfold cross val
                kf = KFold(n_splits=5, random_state=42, shuffle=True)
                cval_r2_scores : list = field(default_factory=list)
                train_r2_scores : list = field(default_factory=list)
                
                # kfold training for each model
                for train_idx, val_idx in kf.split(x_train, y_train):
                    
                    # Split train into train_fold + val_fold
                    x_train_fold, y_train_fold = x_train[train_idx], y_train[train_idx] 
                    x_val_fold, y_val_fold = x_train[val_idx], y_train[val_idx]


                    # Train on train_fold → predict on val_fold (test_r2)
                    model.fit(x_train_fold, y_train_fold)
                    y_val_pred = model.predict(x_val_fold)
                    
                    # Train on train_fold → predict on train_fold (train_r2)  
                    y_train_pred = model.predict(x_train_fold)

                
                   # metrics - r2 scores calulation 
                    cval_r2 = r2_score(y_val_fold, y_val_pred)
                    train_r2 = r2_score(y_train_fold, y_train_pred)
                    cval_r2_scores.append(cval_r2)
                    train_r2_scores.append(train_r2)    
                
                    
                # get the metrics for the model report
                cval_r2_mean = np.mean(cval_r2_scores)
                train_r2_mean = np.mean(train_r2_scores)
                overfit_gap = train_r2_mean - cval_r2_mean
                cval_r2_std = np.std(cval_r2_scores)
                
                
                # store in model report
                trained_models_report[MODELS][model_name] = model
                trained_models_report[CVAL_R2_SCORE][model_name] = cval_r2_mean
                trained_models_report[OVERFIT_GAP][model_name] = overfit_gap
                trained_models_report[CVAL_R2_SCORE_STD][model_name] = cval_r2_std  
            
            logging.info(f"Completed the training of models and generated the model report")    
            return trained_models_report
        
        except Exception as e:
            raise CustomException(e) from None
        
            
    def get_best_model_name(self, trained_models:dict, base_r2_score= BASE_R2_SCORE, 
                       base_overfit_gap= BASE_OVERFIT_GAP)-> str:
        """
        Arguments : trained_models = {
            'models' : {'linear_reg' : linear model, ...},
            'cval_r2_score' : {'linear_reg' : .., ...},
            'overfit_gap' : {...},
            'cval_r2_score_std' : {...}
            }
            
        Return : best_model_name : str
        """
        
        try:
            for model_name, r2_score in trained_models['r2_score'].items():
                overfit_gap = trained_models[OVERFIT_GAP][model_name]
                
                # compare base r2 score
                if r2_score > base_r2_score & overfit_gap < base_overfit_gap:
                    base_r2_score = r2_score
                    best_model_name = model_name
                    
            # return best model
            logging.info(f"Returning the best model name from list of trained models")
            return best_model_name

        except Exception as e:
            raise CustomException(e) from None
        
        
    def create_model_trainer_artifact(self, trained_models:dict, best_model_name:str, x_test:np.ndarray,
                                      y_test: np.ndarray)-> Model_Trainer_Artifact:
        """
        Returns:
            Model_Trainer_Artifact: 
                trained_model_metrics = {'r2_score' : 0.0, 'overfit_gap' : 0.0, 
                'cval_r2_score_std' : 0.0, 'test_r2_score' : 0.0}
        """
        try:
            model = trained_models[MODELS][best_model_name]
            
            test_r2_score = model.score(x_test, y_test)
            
            best_model_metrics = {
                CVAL_R2_SCORE : trained_models[CVAL_R2_SCORE][best_model_name],
                OVERFIT_GAP : trained_models[OVERFIT_GAP][best_model_name],
                CVAL_R2_SCORE_STD : trained_models[CVAL_R2_SCORE_STD][best_model_name],
                TEST_R2_SCORE : test_r2_score
            }
            
            # save model in model file path
            best_model_file_path = self.model_trainer_config.trained_model_file_path
            save_object(object = model, file_path = best_model_file_path)
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
            (x_train, y_train), (x_test, y_test) = self.get_train_and_test_arrays()
            
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
            trained_models_report = self.evaluate_models(x_train=x_train, y_train=y_train, models=models)
            
            # get best model name
            logging.info("Start the best model comparision")
            best_model_name = self.get_best_model_name(trained_models=trained_models_report)
            
            # create model trainer artifact
            logging.info(f"Start the model trainer artifact creation")
            model_trainer_artifact = self.create_model_trainer_artifact(trained_models=trained_models_report,
                                                                        best_model_name=best_model_name, 
                                                                        x_test=x_test, y_test=y_test)
            
            return model_trainer_artifact
        
        except Exception as e:
            raise CustomException(e) from None