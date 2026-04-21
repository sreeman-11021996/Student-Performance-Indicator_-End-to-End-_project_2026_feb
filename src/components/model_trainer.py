# general
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# own methods
from src.logger import logging
from src.exception import CustomException
from src.constants import *
from src.utils import save_object, get_train_and_test_arrays
from src.components.data_transformation import Data_Transformation_Artifact


# metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate



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
    trained_model_metrics = {'val_r2_score' : 0.0, 'overfit_gap' : 0.0, 'val_r2_std' : 0.0, 
    'test_r2_score' : 0.0}
    field(default_factory= lambda : defaultdict(float)) == gives me a new dictionary with float values 
    every new instance.
    """
    trained_model_name : str
    trained_model_file_path : str
    trained_model_metrics : Dict[str, float] = field(default_factory= lambda : defaultdict(float))
    

@dataclass
class Best_Model:
    """
    untrained_model : Ex. {sklearn.tree.RandomForest(n_estimator=100, ...best_parameters...)}
    metrics : {'val_r2_score' : 0.0, 'overfit_gap' : 0.0, 'val_r2_std' : 0.0}
    """
    model_name : str
    untrained_model : Any 
    metrics : dict = field(default_factory= lambda : defaultdict(float))

 
    
class Model_Trainer:
    def __init__(self, data_transformation_artifact:Data_Transformation_Artifact):
        self.data_transformation_arifact = data_transformation_artifact
        self.model_trainer_config = Model_Trainer_Config()
    

# 1. Calculating the Best_Model List    
    @staticmethod
    def kfold_calculate_metrics(x_train:np.ndarray, y_train:np.ndarray, models:dict)-> List[Best_Model]:
        """
        Arguments : models = {'linear_reg' : LinearRegression(), ...}
            
        Return : untrained_model_list = [Best_Model]
                >>> Best_Model = (model_name, untrained_model {like DecisionTreeRegressor()}, metrics dictionary)
        """
        try:
            untrained_model_list: List[Best_Model] = []    
            logging.info(f"Starting the kfold cross validation on models : {models.keys()}")
            # loop all the models
            for model_name, model in models.items():
                
                # save the untrained model
                untrained_model = model
                    
                # kfold cross validation
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
    
                    
                # create the Best_Model instance for th model
                best_model = Best_Model(
                    model_name = model_name,
                    untrained_model = untrained_model,
                    metrics = {VAL_R2_KEY: cval_r2_mean, OVERFIT_GAP_KEY: overfit_gap, VAL_R2_STD_KEY: cval_r2_std}, 
                ) 

                untrained_model_list.append(best_model)    
                logging.info(f"\nmodel name : {best_model.model_name}"
                            f"\nmodel metrics : {best_model.metrics}"
                            f"\nuntrained model : {best_model.untrained_model}")  
                
            logging.info(f"Completed the training of models and generated the model report")    
            return untrained_model_list
            
        except Exception as e:
            raise CustomException(e) from None    

    @staticmethod
    def grid_search_calculate_best_metrics(x_train:np.ndarray, y_train: np.ndarray, models:dict, params:dict)-> List[Best_Model]:
        """
        Summary : cross validation metrics for the best prameters of the model computed using grid search cv
        
        Return : List[Best_Model]
                 >>> Best_Model = (model_name, untrained model set with best parameters from grid search, 
                                  best metrics from grid search) 
        """
        try:            
            logging.info(f"Starting the kfold cross validation on models : {models.keys()}")
            best_model_list: List[Best_Model] = []


            # loop all the models
            for model_name, model in models.items():
                
                # Do grid search for the model
                param = params[model_name]
                grid_cv = GridSearchCV(estimator=model, param_grid=param, cv=5, return_train_score=True)
                grid_cv.fit(x_train,y_train)
                
                # get the best parameters 
                best_parameters = grid_cv.best_params_
                
                # get the best metrics
                best_score_index = grid_cv.best_index_
                best_test_r2 = grid_cv.cv_results_[MEAN_TEST_R2_KEY][best_score_index]
                best_train_r2 = grid_cv.cv_results_[MEAN_TRAIN_R2_KEY][best_score_index]
                best_test_r2_std = grid_cv.cv_results_[STD_TEST_R2_KEY][best_score_index]
                best_overfit_gap = best_train_r2 - best_test_r2
                
                # set the best model
                model.set_params(**best_parameters)
                
                # assign the values to the trained best model report
                best_model = Best_Model(
                    model_name = model_name,
                    untrained_model = model,
                    metrics = {VAL_R2_KEY : best_test_r2, OVERFIT_GAP_KEY : best_overfit_gap, VAL_R2_STD_KEY : best_test_r2_std}
                )             

                best_model_list.append(best_model)   
            
            return best_model_list
        
        except Exception as e:
            raise CustomException(e) from None



# 2. Get me the model with the best r2 score
    @staticmethod
    def get_best_model(best_model_list:List[Best_Model], base_r2_score= BASE_R2_SCORE, 
                       base_overfit_gap= BASE_OVERFIT_GAP)->Optional[Best_Model]:
        try:
            best_r2 = base_r2_score
            best_overfit = base_overfit_gap
            best_model: Optional[Best_Model] = None
            
            for model in best_model_list:
                
                model_r2 = model.metrics[VAL_R2_KEY]
                model_overfit = model.metrics[OVERFIT_GAP_KEY]
            
                if (model_r2 > best_r2) or ((model_r2 == best_r2) and (model_overfit < best_overfit)):
                    best_model = model
                    best_r2 = model_r2
                    best_overfit = model_overfit
                    
            if best_model is None:
                logging.info(f"Empty Best Model. No model has r2 > {base_r2_score}")
  
            return best_model
        
        except Exception as e:
            raise CustomException(e) from None

            
            
# 3. Returns Model Trainer Artifact 
    def create_model_trainer_artifact_new(self, best_model:Optional[Best_Model], x_train:np.ndarray, y_train:np.ndarray, 
                                      x_test:np.ndarray, y_test: np.ndarray)-> Model_Trainer_Artifact:
        try:
            
            # 1. create the model.pkl file dir
            model_trainer_dir = self.model_trainer_config.artifact_dir 
            os.makedirs(model_trainer_dir, exist_ok=True)
            model_file_path = self.model_trainer_config.trained_model_file_path


            # 2. check for empty best model
            if best_model is None:
                model_trainer_artifact = Model_Trainer_Artifact(
                    trained_model_name = "",
                    trained_model_file_path = model_file_path,
                    trained_model_metrics = {}
                )
                logging.info(f"No best model provided → Empty artifact")
                return model_trainer_artifact
            
            
            # 3. train the best model
            model = best_model.untrained_model
            model.fit(x_train, y_train)
            
            # 4. get the metrics & model name
            model_name = best_model.model_name
            test_r2 = model.score(x_test, y_test)
            model_metrics = {
                VAL_R2_KEY : best_model.metrics[VAL_R2_KEY],
                OVERFIT_GAP_KEY : best_model.metrics[OVERFIT_GAP_KEY],
                VAL_R2_STD_KEY : best_model.metrics[VAL_R2_STD_KEY],
                TEST_R2_SCORE_KEY : test_r2
            }
        
            # 5. save the model
            save_object(object = best_model, file_path = model_file_path)
            logging.info(f"Saved the model in {model_file_path}")
            
            # 6. create the model trainer artifact
            model_trainer_artifact = Model_Trainer_Artifact(
                trained_model_name = model_name,
                trained_model_file_path = model_file_path,
                trained_model_metrics = model_metrics
            ) 
            logging.info(f"Model Trainer Artifact Created!")
            
            return model_trainer_artifact
        
        except Exception as e:
            raise CustomException(e) from None
        
               
        
    def initiate_model_trainer(self)-> Model_Trainer_Artifact:
        try:
            logging.info(f"Initaited the model trainer component")
            
            # 1. get the train and test
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
            
            
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "Linear Regression":{},
                "Ridge Regression": {},
                "Lasso Regression": {},
                "k-Neighbours Regression": {},
                
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            
            # Get the List of Best Models 
            logging.info(f"Start evaluation of all the models")
            best_models_list: List[Best_Model] = self.grid_search_calculate_best_metrics(x_train=x_train, y_train=y_train, 
                                                                              models=models, params=params)
            
            # get best model name
            logging.info("Start the best model comparision")
            best_model: Optional[Best_Model] = self.get_best_model(best_model_list=best_models_list)
            
            # create model trainer artifact
            logging.info(f"Start the model trainer artifact creation")
            model_trainer_artifact = self.create_model_trainer_artifact_new(best_model= best_model, x_train=x_train, 
                                                                            y_train=y_train, x_test=x_test, y_test=y_test)
            
            return model_trainer_artifact
        
        except Exception as e:
            raise CustomException(e) from None
        
        