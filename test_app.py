from src.components.data_ingestion import Data_Ingestion
from src.components.data_transformation import Data_Transformation
from src.components.model_trainer import Model_Trainer
from src.components.model_factory import Model_Factory

from src.utils import get_train_and_test_arrays



if __name__ == "__main__":
      data_ingestion = Data_Ingestion()
      data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

      print("Data Ingestion Completed \n"
            f"Train Data Path : {data_ingestion_artifact.train_data_path}\n"
            f"Test Data Path : {data_ingestion_artifact.test_data_path}\n\n")
    
      data_transformation = Data_Transformation(data_ingestion_artifact=data_ingestion_artifact)
      data_transformation_artifact = data_transformation.initiate_data_transforamtion()
    
      print("Data Transformation Completed \n"
            f"Preprocessed Object File Path : {data_transformation_artifact.preprocessed_obj_file_path}\n"
            f"Train Array File Path : {data_transformation_artifact.train_arr_file_path}\n"
            f"Test Array File Path : {data_transformation_artifact.test_arr_file_path}\n\n")
    
      """model_trainer = Model_Trainer(data_transformation_artifact=data_transformation_artifact)
      model_trainer_artifact = model_trainer.initiate_model_trainer()
    
      print("Model Trainer Completed \n"
            f"Trained Model Name : {model_trainer_artifact.trained_model_name}\n"
            f"Trained Model File Path : {model_trainer_artifact.trained_model_file_path}\n"
            f"Trained model metrics : {model_trainer_artifact.trained_model_metrics}\n\n")"""
      
      train_file_path = data_transformation_artifact.train_arr_file_path
      test_file_path = data_transformation_artifact.test_arr_file_path
      (x_train, y_train), (x_test, y_test) = get_train_and_test_arrays(
                                                    train_file_path=train_file_path, 
                                                    test_file_path=test_file_path)
    
      
      # ** debugging from logs : catboost model not working!
      model_config_filepath = "config\model_cat.yaml"
      model_factory = Model_Factory(model_config_file_path=model_config_filepath)
      model_factory.initiate_model_factory(input_feature=x_train, output_feature=y_train)
      print("Modle Factory Completed \n"
            f"List of best models : \n")
      for best_model in model_factory.Grid_Searched_Best_Models_List:
            print(f"Model Details : {best_model.model_detail}\n"
                  f"Fine Tuned Model : {best_model.tuned_model}\n"
                  f"Best Grid Searched Parameters : {best_model.best_parameters}\n"
                  f"Metrics : {best_model.metrics}\n\n")