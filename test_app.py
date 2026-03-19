from src.components.data_ingestion import Data_Ingestion
from src.components.data_transformation import Data_Transformation



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