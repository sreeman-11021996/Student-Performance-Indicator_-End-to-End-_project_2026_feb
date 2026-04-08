# general
import os
from typing import Dict, List
from dataclasses import dataclass, field
from collections import defaultdict

# own methods
from src.logger import logging
from src.exception import CustomException
from src.constants import *
from src.utils import save_object, save_numpy_array_data
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
import pandas as pd



@dataclass
class Model_Trainer_Config:
    artifact_dir : str = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR)
    trained_model_file_path : str = os.path.join(artifact_dir, TRAINED_MODEL_FILENAME)
    

@dataclass
class Model_Trainer_Artifact:
    """
    trained_model_metrics = {'r2_score' : 0.0, 'overfit_gap' : 0.0, 'rmse' : 0.0, 
    'cval_r2_score_std' : 0.0}
    """
    trained_model_name : str
    trained_model_file_path : str
    trained_model_metrics : Dict[str, float] = field(default_factory= lambda : defaultdict(float))
    

 
    
class Model_Trainer:
    def __init__(self, data_transformation_artifact:Data_Transformation_Artifact):
        self.data_transformation_arifact = self.data_transformation_arifact
        self.model_trainer_config = Model_Trainer_Config()
        
    
    def evaluate_model(self, x_train:np.ndarray, y_train:np.ndarray, models:dict)-> Dict:
        """
        Arguments : models = {'linear_reg' : LinearRegression(), ...}
        
        Return : trained_models = {
            'models' : {'linear_reg' : linear model, ...},
            'r2_score' : {'linear_reg' : .., ...},
            'overfit_gap' : {...},
            'rmse' : ..., 
            'cval_r2_score_std' : ...
            }
        """
        try:
            # set up the report dictionary
            trained_model_report_fields = [MODELS, CVAL_R2_SCORE, OVERFIT_GAP, RMSE, CVAL_R2_SCORE_STD]
            trained_model_report = dict.fromkeys(trained_model_report_fields, {})
            
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
                trained_model_report[MODELS][model_name] = model
                trained_model_report[CVAL_R2_SCORE][model_name] = cval_r2_mean
                trained_model_report[OVERFIT_GAP][model_name] = overfit_gap
                trained_model_report[CVAL_R2_SCORE_STD][model_name] = cval_r2_std  
                
            return trained_model_report
        
        except Exception as e:
            raise CustomException(e) from None
        
        
    
    def get_best_model(self, trained_models:dict, base_r2_score= BASE_R2_SCORE, 
                       overfit_gap= OVERFIT_GAP)-> Dict:
        """
        Arguments : trained_models = {
            'models' : {'linear_reg' : linear model, ...},
            'r2_score' : {'linear_reg' : .., ...},
            'overfit_gap' : {...},
            ...
            }
            
        Return : best_model = {
            'model_name' : str, 'model' : model, 
            'metrics' : {'r2_score' : 0.0, 'overfit_gap' : 0.0, 'rmse' : 0.0, 'cval_r2_score_std' : 0.0} 
            }
        """
        return {}
