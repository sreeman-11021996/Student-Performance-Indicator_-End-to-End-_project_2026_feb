import os
from typing import Any
from collections import defaultdict

from src.exception import CustomException
from src.logger import logging
from src.constants import *


import pandas as pd
import numpy as np
import dill

# metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_validate



def get_categorical_columns (data:pd.DataFrame)->list[str]:
    """Return a list of column names that are categorical."""
    try:
        cat_cols = data.select_dtypes(include='object').columns
        return cat_cols
    
    except Exception as e:
        raise CustomException(e) from None



def get_numerical_columns(data:pd.DataFrame)->list[str]:
    """Return a list of column names that are numerical."""
    try:
        num_cols = data.select_dtypes(exclude='object').columns
        return num_cols
    
    except Exception as e:
        raise CustomException(e) from None
 
 
    
def get_X_and_y (data:pd.DataFrame, target_column:str)->tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=[target_column],axis=1)
        Y = data[target_column]
        return X, Y
    
    except Exception as e:
        raise CustomException(e) from None



def save_object(object:object, file_path:str):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        # write to the file in binary to keep the 
        with open(file_path, "wb") as file_obj:
            dill.dump(object, file_obj)
    
    except Exception as e:
        raise CustomException(e) from None
    
    
    
def load_object(file_path:str)-> Any:
    try:
        # read the file in binary
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e) from None
    
    
    
    
def save_numpy_array_data(np_array:np.ndarray, file_path:str):
    """
    np.save(file_path, array)
    """
    try:

        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        # write to the file in binary to keep the 
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, np_array)
    
    except Exception as e:
        raise CustomException(e) from None
    
    
    
def load_numpy_array_data(file_path:str)-> np.ndarray:
    try:
        # read the file in binary
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    
    except Exception as e:
        raise CustomException(e) from None
    

def get_train_and_test_arrays (train_file_path:str, test_file_path:str)->tuple:
    """
    Returns:
    tuple: (x_train, y_train), (x_test, y_test) : all are np.ndarrays
    """
    try:            
        train_arr = load_numpy_array_data(file_path = train_file_path)
        test_arr = load_numpy_array_data(file_path = test_file_path)
        logging.info(f"Loaded the train & test arrays from artifacts")
            
        x_train, y_train, x_test, y_test = (train_arr[:,:-1], train_arr[:,-1], 
                                                test_arr[:,:-1], test_arr[:,-1])
            
        return (x_train, y_train), (x_test, y_test)    
    
    except Exception as e:
        raise CustomException(e) from None

    
def evaluate_models(x_train:np.ndarray, y_train:np.ndarray, models:dict)-> Dict:
    """
    Arguments : models = {'linear_reg' : LinearRegression(), ...}
        
    Return : trained_models_report = {
        'cval_r2_score' : {'linear_reg' : .., ...},
        'overfit_gap' : {...},
        'cval_r2_score_std' : ...
        }
    """
    try:
        # set up the report dictionary
        trained_models_report = defaultdict(lambda: defaultdict(float))
            
        logging.info(f"Starting the kfold cross validation on models : {models.keys()}")
        # loop all the models
        for model_name, model in models.items():
                 
            # kfold cross val
            kf = KFold(n_splits=5, random_state=42, shuffle=True)
            cval_r2_scores : list = []
            train_r2_scores : list = []
                
            # get r2 scores using kfold validation for each model
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
 
                
            # store metrics in model report
            trained_models_report[CVAL_R2_SCORE][model_name] = float(cval_r2_mean)
            trained_models_report[OVERFIT_GAP][model_name] = float(overfit_gap)
            trained_models_report[CVAL_R2_SCORE_STD][model_name] = float(cval_r2_std)
                
            logging.info(f"\nmodel name : {model_name}"
                         f"\ncval r2 mean : {cval_r2_mean}, {trained_models_report[CVAL_R2_SCORE][model_name]}"
                         f"\noverfit gap : {overfit_gap}, {trained_models_report[OVERFIT_GAP][model_name]}"
                         f"\ncval r2 std : {cval_r2_std}, {trained_models_report[CVAL_R2_SCORE_STD][model_name]}")  
            
        logging.info(f"Completed the training of models and generated the model report")    
        return trained_models_report
        
    except Exception as e:
        raise CustomException(e) from None



def get_best_model_name (trained_models:dict, base_r2_score= BASE_R2_SCORE, 
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
        best_model_name : str = ''
        base_r2_score_local = base_r2_score
            
        for model_name, r2_score in trained_models[CVAL_R2_SCORE].items():    
            overfit_gap = trained_models[OVERFIT_GAP][model_name]
                
            # compare base r2 score
            if r2_score > base_r2_score_local and overfit_gap < base_overfit_gap:
                base_r2_score_local = r2_score
                best_model_name = model_name
                    
        # return best model
        logging.info(f"Returning the best model name from list of trained models")
        logging.info(f"The Best Trained Model : {best_model_name}")
        return best_model_name

    except Exception as e:
        raise CustomException(e) from e
