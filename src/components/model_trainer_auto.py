# general
import os
from typing import List
from dataclasses import dataclass

# own methods
from src.logger import logging
from src.exception import CustomException
from src.constants import *
from src.utils import save_object, get_train_and_test_arrays
from src.components.data_transformation import Data_Transformation_Artifact
from src.model_factory import Model_Factory, Best_Model 

# data operation
import numpy as np



@dataclass
class Model_Trainer_Config:
    artifact_dir : str = os.path.join(ARTIFACT_DIR, CURRENT_TIME_STAMP, MODEL_TRAINER_DIR)
    trained_model_file_path : str = os.path.join(artifact_dir, TRAINED_MODEL_FILENAME)
    

@dataclass
class Model_Trainer_Artifact:
    """
    trained_model_name : (Ex. "DecisonTree")
    trained_model_file_path : the file which stores the trained model (Ex. ../../model.pkl)
    trained_model_metrics : {'val_r2_score' : value, 'val_r2_std' : value, 'overfit_gap' : value, 'test_r2_score' : val}
    """
    trained_model_name : str
    trained_model_file_path : str
    trained_model_metrics: dict

 
    
class Model_Trainer:
    def __init__(self, data_transformation_artifact:Data_Transformation_Artifact):
        self.data_transformation_arifact = data_transformation_artifact
        self.model_trainer_config = Model_Trainer_Config()


    @staticmethod
    def get_best_model (best_models_list:List[Best_Model])-> Best_Model:
        """
        Arguments : 
            best_models : List [Best_Model]

                        
        Return : 
            Best_Model
                        - tuned_model : grid searched model with best parameters (Untrained)
                        - model_detail : dict {'model_serial_number' : 'model_0', 'model_name' : "DecisionTree"}           
                        - best_parameters = {'max_depth':4?,'min_leaf':value..} 
                                            grid searched best parameters for the model type (ex. decision tree)
                        - metrics = {'val_r2_score' : value, 'val_r2_std' : value, 'overfit_gap' : value}            
        """
        try:
            # 1. initial best
            best_r2_score = None
                        
            # 2. get the best model via comparision of r2 score and overfit gap
            for model in best_models_list:
                # 2.a check : empty model   
                if model.best_parameters == {}:
                    logging.info(f"The model {model.model_detail[MODEL_NAME_KEY]} "
                                 f"has no best parameters that can be trained to get a R2 > 0.6")
                    continue
                
                
                # 2.b assign the checking parameters
                r2_score = model.metrics[VAL_R2_KEY]    
                overfit_gap = model.metrics[OVERFIT_GAP_KEY]
                     
                
                # 2.c check : greater r2 score wins (or) for same r2 score, lower overfit gap wins
                if ((best_r2_score is None) or (r2_score > best_r2_score)) or ((r2_score == best_r2_score) and 
                                                                               (overfit_gap < best_overfit_gap)):
                    best_r2_score = r2_score
                    best_overfit_gap = overfit_gap
                    best_model = model

                        
            # 3. return best model
            logging.info(f"The Best Model with Grid Search best parameters : {best_model.model_detail[MODEL_NAME_KEY]}\n"
                         f"------------------------\n"
                         f"Best Model Details : {best_model.model_detail}\n"
                         f"Best Parameters : {best_model.best_parameters}\n"
                         f"Best Model validation r2 : {best_model.metrics[VAL_R2_KEY]:.4f}\n"
                         f"Best Model Overfit Gap : {best_model.metrics[OVERFIT_GAP_KEY]:.4f}\n"
                         f"------------------------\n"
                         )
            return best_model

        except Exception as e:
            raise CustomException(e) from None

         
    def create_model_trainer_artifact(self, best_model:Best_Model, train_input_feature:np.ndarray, train_output_feature:np.ndarray,
                                      test_input_feature:np.ndarray, test_output_feature:np.ndarray)-> Model_Trainer_Artifact:
        """  
        Arguments :
            best_model : Best_Model (tuned_model, model_detail, best_parameters, metrics) 
                                     tuned_model = .fit/.predict/.score()
                                     metrics : {'val_r2_score' : val, 'val_r2_std' : val, 'overfit_gap' : val}
        
        Returns:
            Model_Trainer_Artifact: 
                trained_model_name = "DecisionTree"
                trained_model_file_path = "/../../.pkl"
                trained_model_metrics : {'val_r2_score' : value, 'val_r2_std' : value, 'overfit_gap' : value, 
                                        'test_r2_score' : val}
        """
        try:
            
            # 1. Train the best model
            model = best_model.tuned_model
            model.fit(train_input_feature, train_output_feature)
            
            # 2. Assign the artifact arguments (model name, metrics)
            model_name = best_model.model_detail[MODEL_NAME_KEY]
            test_r2_score = model.score(test_input_feature, test_output_feature)
            model_metrics = {
                VAL_R2_KEY : best_model.metrics[VAL_R2_KEY],
                OVERFIT_GAP_KEY : best_model.metrics[OVERFIT_GAP_KEY],
                VAL_R2_STD_KEY : best_model.metrics[VAL_R2_STD_KEY],
                TEST_R2_SCORE_KEY : test_r2_score
            }
            
            # 3. save model in model file path
            model_file_path = self.model_trainer_config.trained_model_file_path
            save_object(object = model, file_path = model_file_path)
            logging.info(f"Saved the {model_name} model in file : {model_file_path}")
            
            # 4. create model trainer artifact
            model_trainer_artifact = Model_Trainer_Artifact(
                trained_model_name = model_name,
                trained_model_file_path = model_file_path,
                trained_model_metrics = model_metrics
            )
            logging.info("Created the Model Trainer Artifact") 
            
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
            
            
            # 2. get the list of models with the best parameters grid searched
            logging.info(f"Run the Model Factory to get the list of Best Models (Grid Searched) List")
            model_config_filepath = os.path.join(CONFIG_DIR,MODEL_CONFIG_FILE_NAME)
            model_factory = Model_Factory(model_config_file_path=model_config_filepath)
            
            best_models:List[Best_Model] = model_factory.initiate_model_factory(input_feature=x_train, output_feature=y_train)
            
            
            # 3. Select the best model from the best_models list          
            logging.info("Start the best model comparision")
            best_model: Best_Model = self.get_best_model(best_models_list=best_models)
            
            
            # 4. create model trainer artifact
            logging.info(f"Start the model trainer artifact creation")
            model_trainer_artifact = self.create_model_trainer_artifact(best_model=best_model,train_input_feature=x_train, 
                                            train_output_feature=y_train, test_input_feature=x_test, test_output_feature=y_test)
            
            return model_trainer_artifact
        
        except Exception as e:
            raise CustomException(e) from None
        
        