import os
from typing import Any

from src.exception import CustomException
from src.logger import logging


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