import os
from typing import Any
from collections import defaultdict

from src.exception import CustomException
from src.logger import logging
from src.constants import *


import numpy as np
import dill


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

    


