import os
from typing import Any

from src.exception import CustomException
from src.logger import logging
from src.constants import *


import pandas as pd
import numpy as np
import dill



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
    

def get_train_and_test_arrays(train_file_path:str, test_file_path:str)->tuple:
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

    

def get_best_model_name(trained_models:dict, base_r2_score= BASE_R2_SCORE, 
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
