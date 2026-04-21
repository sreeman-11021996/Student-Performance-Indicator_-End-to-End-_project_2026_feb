from src.components.data_ingestion import Data_Ingestion
from src.components.data_transformation import Data_Transformation
from src.components.model_trainer_auto import Model_Trainer 



if __name__ == "__main__":
      
      # 1. Data Ingestion
      data_ingestion = Data_Ingestion()
      data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

      print("Data Ingestion Completed \n"
            f"Train Data Path : {data_ingestion_artifact.train_data_path}\n"
            f"Test Data Path : {data_ingestion_artifact.test_data_path}\n\n")
    
    
    
      # 2. Data Transformation
      data_transformation = Data_Transformation(data_ingestion_artifact=data_ingestion_artifact)
      data_transformation_artifact = data_transformation.initiate_data_transforamtion()
    
      print("Data Transformation Completed \n"
            f"Preprocessed Object File Path : {data_transformation_artifact.preprocessed_obj_file_path}\n"
            f"Train Array File Path : {data_transformation_artifact.train_arr_file_path}\n"
            f"Test Array File Path : {data_transformation_artifact.test_arr_file_path}\n\n")
 
 
      # ** Verify the best r2 from model trainer vs model trainer auto
      # 3. Model Training 
      model_trainer = Model_Trainer(data_transformation_artifact=data_transformation_artifact)
      model_trainer_artifact = model_trainer.initiate_model_trainer()
    
      print("Model Trainer Completed \n"
            f"Trained Model Name : {model_trainer_artifact.trained_model_name}\n"
            f"Trained Model File Path : {model_trainer_artifact.trained_model_file_path}\n"
            f"Trained model metrics : {model_trainer_artifact.trained_model_metrics}\n\n")


      
      
      # 4. Model Factory

      # ** CHange src.constants.py for production!!!!
    