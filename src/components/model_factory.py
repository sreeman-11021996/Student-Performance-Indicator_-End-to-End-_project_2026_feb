import os
from collections import defaultdict
from typing import Any, List, Tuple, Optional, Dict

from src.exception import CustomException
from src.logger import logging
from src.constants import *

import numpy as np
from dataclasses import dataclass, field
import yaml

# models
import importlib



def get_sample_model_config_yaml_file(export_dir:str):
    
    try:
        # Sample configuration structure
        model_config = {
            GRID_SEARCH_KEY: {
                MODULE_KEY: "sklearn.model_selection",
                CLASS_KEY: "GridSearchCV",
                PARAM_KEY: {
                    "cv": 3,        # 3-fold cross-validation
                    "verbose": 1    # Show progress during search
                }
            },
            MODEL_SELECTION_KEY: {
                "module_0": {   # First model to test
                    MODULE_KEY: "module_of_model",      # Replace with actual module
                    CLASS_KEY: "ModelClassName",        # Replace with actual class
                    PARAM_KEY: {
                        "param_name1": "value1",
                        "param_name2": "value2",
                    },
                    SEARCH_PARAM_GRID_KEY: {
                        "param_name": ['param_value_1', 'param_value_2']
                    }
                },
            }
        }
        
        # make directory
        os.makedirs(export_dir, exist_ok=True)
        
        # config file path
        export_file_path = os.path.join(export_dir, MODEL_CONFIG_FILENAME)
        
        # save model dict in file
        with open(export_file_path, 'w') as file_obj:
            yaml.dump(model_config, file_obj, default_flow_style=False)
            
    except Exception as e:
        raise CustomException(e) from e



@dataclass
class Untuned_Model:
    """
    model: Instantiated scikit-learn model object
    model_detail : dict {'model_serial_number' : 'model_0', 'model_name' : "..."}
                    model_serial_number: ID like "model_0"
                    model_name: String like "sklearn.tree.DecisionTreeRegressor"
    grid_search_parameters: Dictionary of hyperparameters to search
    """
    model : Any
    model_detail : dict = field(default_factory= dict)
    grid_search_parameters : dict = field(default_factory=dict)


@dataclass
class Grid_Searched_Model:
    """ 
    model_serial_number : 'model_0'
    best_parameters = grid searched best parameters for the model type (ex. decision tree)
    metrics = {'val_r2_score' : val, 'val_r2_std' : val, 'overfit_gap' : val}
    """
    model_serial_number : str
    parameters : dict = field(default_factory=dict)
    metrics : dict = field(default_factory= lambda: defaultdict(float))


@dataclass
class Best_Model:
    """
    tuned_model : grid searched model with best parameters (Untrained)
    model_detail : dict {'model_serial_number' : 'model_0', 'model_name' : "..."}
                    model_serial_number: ID like "model_0"
                    model_name: String like "sklearn.tree.DecisionTreeRegressor"    
    best_parameters = grid searched best parameters for the model type (ex. decision tree)
    metrics = {'val_r2_score' : val, 'val_r2_std' : val, 'overfit_gap' : val}
    """
    tuned_model : Any
    model_detail : Dict[str,str] = field(default_factory= lambda: defaultdict(str))
    best_parameters : dict = field(default_factory=defaultdict)
    metrics : dict = field(default_factory= lambda: defaultdict(float))
    
    
# ** optimization to be done
class Model_Factory:
    
    def __init__(self, model_config_file_path:str):
        try:
            self.model_config = self.read_config_yaml_file(file_path=model_config_file_path)
            
            # 1. initialize grid search details
            self.grid_search_details: dict = self.model_config[GRID_SEARCH_KEY]
            
            # 2. initalize untuned model details
            self.models_details: dict = self.model_config[MODEL_SELECTION_KEY]
            
            # 3. initialize the untuned models lists 
            self.Untuned_Models_List: List[Untuned_Model] = []
             
        except Exception as e:
            raise CustomException(e) from e
   
   
   
   # 1. functionality needed in class
    @staticmethod
    def read_config_yaml_file(file_path:str)->dict:
        try:
            if file_path is None:
                raise ValueError("Config path is given as None")
            
            with open(file_path, 'r',  encoding='utf-8') as yaml_file_obj:
                model_config = yaml.safe_load(yaml_file_obj)
            
            return model_config
        
        except Exception as e:
            raise CustomException(e) from e
    
    @staticmethod
    def get_model_class_reference(module_name:str, class_name:str)->Any:
        """
        Dynamically import class from string

        Returns:
            Any: Example. <class sklearn.model_selection.DecisionTreeRegressor> class reference 
        """
        try:
            module = importlib.import_module(module_name)
            class_reference = getattr(module, class_name)
        
            return class_reference
        
        except Exception as e:
            raise CustomException(e) from e
        
    @staticmethod
    def set_model_class_properties(model_obj:Any, property_data:dict)-> Any:
        """
        Set the parameters for the model object (instance_ref)

        Args:
            model_obj (Any): Example. DecisionTreeRegressor()
            property_data (dict): {'criterion' : 'squared_error', 'min_samples_leaf' : 2,
            'max_depth' : [2,3,4,5,6,7,8,9]}
        """
        try:
            if not isinstance(property_data, dict):
                raise Exception("property_data parameter required to be dictionary")
            
            # safe parameters setting - for sklearn and other models
            model_obj.set_params(**property_data)

            return model_obj
            
        except Exception as e:
            raise CustomException(e) from e
        
   
    # 2. creating untuned models list
    def create_untuned_model(self, model_number:str, model_config:dict)->Untuned_Model:
        """
        Create single untuned model from config dict.
    
        Args:
            model_config: {'module': 'sklearn.tree', 'class': 'DecisionTreeRegressor', ...}
    
        Returns:
            Fully configured Untuned_Model instance
        """
        try:            
            # 1. Get model class → instantiate
            module_name = model_config[MODULE_KEY]
            class_name = model_config[CLASS_KEY]
            model_class = self.get_model_class_reference(module_name=module_name, class_name=class_name)
            model = model_class()            
            
            # 2. Set fixed parameters
            if PARAM_KEY in model_config:
                model_property = model_config[PARAM_KEY]  
                model = self.set_model_class_properties(model_obj=model, property_data=model_property)
            
            # 3. Get grid search parameters 
            model_grid_search_parameters = model_config[SEARCH_PARAM_GRID_KEY]
            
            # 4. Get the model details
            model_name = model.__class__.__name__
            model_detail={MODEL_NAME_KEY:model_name,MODEL_NUMBER_KEY:model_number}
            
            # 4. Create untuned model
            untuned_model = Untuned_Model(
                model = model,
                grid_search_parameters = model_grid_search_parameters,
                model_detail = model_detail
                )
            logging.info(f"Created untuned model : {model_detail[MODEL_NAME_KEY]}")
            return untuned_model
                 
        except Exception as e:
            raise CustomException(e) from e
        
    def initiate_untuned_models_list(self)->None:
        try:
            logging.info(f"Starting to initiate the untuned models")
            for model_number, model_config in self.models_details.items():
                
                # 1. create an untuned model instance 
                untuned_model = self.create_untuned_model(model_number=model_number, 
                                                          model_config=model_config)
                
                # 2. append to untuned models list
                self.Untuned_Models_List.append(untuned_model)
                
            logging.info(f"Untuned Models List : {[model.model_detail for model in self.Untuned_Models_List]}\n")

            
        except Exception as e:
            raise CustomException(e) from e  
        


    # 3. grid search cv list
    def grid_search_tuning_model(self, untuned_model:Untuned_Model, input_feature:np.ndarray,
                               output_feature:np.ndarray)->Tuple[str,dict]:
        """
        Compute the Grid Search CV on a single untuned model 
    
        Args:
            untuned_model: Untuned_Mode(model, model_detail, grid_search_parameters)
    
        Returns: (model_name, grid_search_result)
            A Tuple of model_name and grid search results dict containing the metrics and their parameters
        """
        try:
            # Logging the start
            model_name = untuned_model.model_detail[MODEL_NAME_KEY]
            logging.info(f"Start Grid Search CV for : {model_name}")
            
            # 1. Get grid search class → instantiate : GridSearchCV()
            module_name = self.grid_search_details[MODULE_KEY]
            class_name = self.grid_search_details[CLASS_KEY]
            model_class = self.get_model_class_reference(module_name=module_name, class_name=class_name)
            
            
            # 2. Create grid search instance : {estimator : model, params : dict}
            estimator = untuned_model.model
            grid_search_parameters = untuned_model.grid_search_parameters
            base_grid_search = model_class(estimator=estimator,param_grid=grid_search_parameters)  
            
            
            # 3. Set fixed grid search parameters : {cv, verbose, n_jobs}
            grid_search_property: dict = self.grid_search_details[PARAM_KEY]  
            grid_search_cv = self.set_model_class_properties(model_obj=base_grid_search, property_data=grid_search_property)            
                        
                        
            # 5. Train grid search cv
            grid_search_cv.fit(input_feature, output_feature)
          
            
            # 6. Get the result from grid search cv with metrics and parameters
            grid_search_result: dict = grid_search_cv.cv_results_
            logging.info(f"Completed Grid Search CV on model : {model_name}")
                        
            return (model_name, grid_search_result)
        
        except Exception as e:
            raise CustomException(e) from e
        
    def parse_grid_search_cv_results(self, model_name:str, grid_search_result:dict)->List[Grid_Searched_Model]:
        """
        We take the result dictionary and convert it into a list of grid searched model instances

        Returns:
            List[Grid_Searched_Model]: [(best_parameters,metrics),...]
            Grid_Searched_Model : 
                best_parameters : {'C': 1, 'kernel': 'rbf'}
                metrics : {'val_r2_score' : val, 'val_r2_std' : val, 'overfit_gap' : val}
        """
        try:
            # 1. get the metrics and parameters from grid_searc_result
            test_r2_mean = grid_search_result.get(MEAN_TEST_R2_KEY, [])
            train_r2_mean = grid_search_result.get(MEAN_TRAIN_R2_KEY, [])
            test_r2_std = grid_search_result.get(STD_TEST_R2_KEY, [])
            parameters = grid_search_result[PARAM_KEY]
        
            # 2. create Grid Searched Model instance for all the parameter combos 
            grid_search_iter = len(test_r2_mean)
            grid_searched_model_list : List[Grid_Searched_Model] = []

            for iter in range(grid_search_iter):
                model_number = MODEL_NUMBER_STRING_KEY + str(iter)
                grid_searched_model = Grid_Searched_Model(model_serial_number=model_number)

                # metrics
                grid_searched_model.metrics[VAL_R2_KEY] = test_r2_mean[iter]
                grid_searched_model.metrics[VAL_R2_STD_KEY] = test_r2_std[iter]
                grid_searched_model.metrics[OVERFIT_GAP_KEY] = train_r2_mean[iter] - test_r2_mean[iter]

                # parameters
                grid_searched_model.parameters = parameters[iter]

                # append to Grid_Model_List
                grid_searched_model_list.append(grid_searched_model)
            
            logging.info(f"Grid Search CV Results Parsed for : {model_name}")  
                  
            # 3. return the grid searched model list
            return grid_searched_model_list
        
        except Exception as e:
            raise CustomException(e) from e
        
    def grid_search_tuning_models(self, input_feature:np.ndarray, output_feature:np.ndarray)->dict:
        """
        Get the dictonary of grid search cv results 
        
        Return : grid_search_cv_results : dict
                {
                    'model_1' : 
                    {
                        'model_name' : 'DecisionTree',
                        'model' : DecisionTreeRegressor(criterion='squared_error', min_samples_leaf=2), 
                        'grid_search_result_list' : [Grid_Searched_Model instances]
                    }
                }
        """
        try:
            logging.info(f"Starting to Tune the models using Grid Search CV")
            grid_search_cv_results:dict = defaultdict(dict)

            for untuned_model in self.Untuned_Models_List:                
                # 1. compute the grid search cv       
                model_name, grid_search_result = self.grid_search_tuning_model(untuned_model=untuned_model,
                                                                               input_feature=input_feature,
                                                                               output_feature=output_feature)
                
                # 2. get list of Grid_Searched_Model instances
                grid_search_result_list: List[Grid_Searched_Model] = self.parse_grid_search_cv_results(model_name = model_name,
                                                                                           grid_search_result = grid_search_result)                          
                logging.info(f"Created the Grid Searched Models List for Model : {model_name}\n")


                # 3. get the required keys for grid_search_cv_results dictionary
                model_number = untuned_model.model_detail[MODEL_NUMBER_KEY]


                # 4. populate the grid_search_cv_results dictionary
                grid_search_cv_results[model_number][MODEL_NAME_KEY] = model_name
                grid_search_cv_results[model_number][MODEL_KEY] = untuned_model.model
                grid_search_cv_results[model_number][GRID_SEARCH_RESULT_LIST_KEY] = grid_search_result_list

                                                
            return grid_search_cv_results   
                
        except Exception as e:
            raise CustomException(e) from e
        
        
        
    # 4. creating best models list 
    def create_best_model(self, model_number:str, grid_search_result:dict, base_r2=BASE_R2, 
                                     overfit_gap=OVERFIT_GAP)->Best_Model:
        """
        ...
        Args:
            model_number (str): 'model_1' 
            grid_search_result (dict): 
                    {
                        'model_name' : 'DecisionTree',
                        'model' : DecisionTreeRegressor(criterion='squared_error', min_samples_leaf=2), 
                        'grid_search_result_list' : [Grid_Searched_Model instances]
                    }
        Returns:
            Best_Model: 
                tuned_model : grid searched model with best parameters
                model_detail : dict {
                                    'model_serial_number' : 'model_0', 
                                    'model_name' : "sklearn.tree.DecisionTreeRegressor"
                                    }
                best_parameters = {'param_1':val_1, ...}
                metrics = {'val_r2_score' : val, 'val_r2_std' : val, 'overfit_gap' : val}
        """
        try:
            # 1. initiate variales required
            local_base_r2 = base_r2
            local_overfit_gap = overfit_gap
            grid_models_list: List[Grid_Searched_Model] = grid_search_result[GRID_SEARCH_RESULT_LIST_KEY]
            best_model: Optional[Grid_Searched_Model] = None 
            
  
            # 2. search for the best model in list 
            for grid_model in grid_models_list:
     
                # get the r2 score and overfit gap of the model
                grid_model_r2 = grid_model.metrics[VAL_R2_KEY]
                grid_model_overfit_gap = grid_model.metrics[OVERFIT_GAP_KEY]
                
        
                # greater r2 score wins (or) for same r2 score, lower overfit gap wins
                if ((grid_model_r2 > local_base_r2) and (grid_model_overfit_gap < overfit_gap)) or (
                    (grid_model_r2 == local_base_r2) and (grid_model_overfit_gap < local_overfit_gap)):
                    
                    best_model = grid_model
                    local_base_r2 = grid_model_r2
                    local_overfit_gap = grid_model_overfit_gap


            # 3. get the common arguments to create a best model
            model_name = grid_search_result[MODEL_NAME_KEY]
            untuned_model = grid_search_result[MODEL_KEY]
            model_detail: dict = {MODEL_NUMBER_KEY: model_number, MODEL_NAME_KEY: grid_search_result[MODEL_NAME_KEY]}   
            
                                 
            # 4. check if no model meets criteria (fall back)
            if best_model is None:
                logging.warning(f"No valid {model_name} (model_no={model_number}): "
                                f"R²≥{base_r2:.3f}, gap≤{overfit_gap:.3f}. Returning empty Best_Model.")
                
                # create an empty best model
                empty_best_model = Best_Model(
                    tuned_model = untuned_model,  
                    model_detail= model_detail,
                    best_parameters={}, 
                    metrics={VAL_R2_KEY: 0.0, VAL_R2_STD_KEY: 0.0, OVERFIT_GAP_KEY: float('inf')}
                )
                logging.info(f"Created empty Best_Model for: {model_name}")
                return empty_best_model
                
                
            # 5. get the Best_model arguments
            best_parameters: dict = best_model.parameters
            tuned_model = self.set_model_class_properties(model_obj=untuned_model,property_data=best_parameters)    
            metrics: dict = best_model.metrics
            
                
            # 6. create the Best_model instance
            best_grid_model = Best_Model(
                tuned_model = tuned_model,
                model_detail = model_detail,
                best_parameters = best_parameters,
                metrics = metrics
            )

            
            logging.info(f"Created the Best Model for : {best_grid_model.model_detail[MODEL_NAME_KEY]}")        
            return best_grid_model    
                    
        except Exception as e:
            raise CustomException(e) from e
        
    def initiate_best_models_list(self, grid_search_cv_results: dict)->List[Best_Model]:
        """Get the best model for each type of model into a list

        Args:
            grid_search_cv_results (dict): 
                                    {
                                        'model_1' : 
                                        {
                                            'model_name' : 'DecisionTree',
                                            'model' : DecisionTreeRegressor(criterion='squared_error', min_samples_leaf=2), 
                                            'grid_search_result_list' : [Grid_Searched_Model instances]
                                        }
                                    }

        Returns:
            List[Best_Model]: _description_
        """
        try:
            logging.info(f"Starting to initialize the best models from the Grid Searched Models foe all the models")
            Grid_Searched_Best_Models_List: List[Best_Model] = []
            
            # compute the best grid models list
            for model_number, grid_search_result in grid_search_cv_results.items():
                
                # 1. get best grid search model
                best_grid_model: Best_Model = self.create_best_model(model_number=model_number,
                                        grid_search_result=grid_search_result)
                
                # 2. add to the list
                Grid_Searched_Best_Models_List.append(best_grid_model)
            
            return Grid_Searched_Best_Models_List
        
        except Exception as e:
            raise CustomException(e) from e



    # 5. main control function
    def initiate_model_factory(self, input_feature:np.ndarray, output_feature:np.ndarray):
        try:
            logging.info(f"Starting the Model Factory")
            
            # 1. set up the list of untuned models
            self.initiate_untuned_models_list()
            logging.info(f"initiated the untuned models list")
            
            # 2. Do Grid Search CV for on untuned models list 
            grid_search_cv_results: dict = self.grid_search_tuning_models(input_feature=input_feature, 
                                                                          output_feature=output_feature)
            logging.info(f"Performed Grid Search CV on all the untuned models")
            
            # 3. Get the best grid search cv models
            best_models_list: List[Best_Model] = self.initiate_best_models_list(grid_search_cv_results=grid_search_cv_results)
            logging.info(f"Computed the best model parameters for each model using grid search cv in a list")
            
            
            # Log the best model list for view
            logging.info(f"The Best Models List : \n")
            for model in best_models_list:
                logging.info(
                    f"------------------------\n"
                    f"Model Details : {model.model_detail}\n"
                    f"Model validation r2 : {model.metrics[VAL_R2_KEY]:.4f}\n"
                    f"Model Overfit Gap : {model.metrics[OVERFIT_GAP_KEY]:.4f}\n"
                    f"------------------------\n"
                )
            
            return best_models_list
        
        except Exception as e:
            raise CustomException(e) from e    