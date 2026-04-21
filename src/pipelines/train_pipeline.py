# components
from src.components.data_ingestion import Data_Ingestion, Data_Ingestion_Artifact
from src.components.data_transformation import Data_Transformation, Data_Transformation_Artifact
from src.components.model_trainer_auto import Model_Trainer, Model_Trainer_Artifact

from src.exception import CustomException
from src.logger import logging

class Training_Pipeline:
    
    def __init__(self) -> None:
        logging.info(f"Starting the Training Pipeline!")
        pass
    

# Training Pipeline Steps
    def start_data_ingestion(self)->Data_Ingestion_Artifact:
        try:
            logging.info(f"Starting the Data Ingesiton")
            data_ingestion = Data_Ingestion()
            data_ingestion_artifact: Data_Ingestion_Artifact = data_ingestion.initiate_data_ingestion()

            logging.info(f"Data Ingestion Artifact Created!")
            return data_ingestion_artifact
        
        except Exception as e:
            raise CustomException(e) from None
        
    """def start_data_validation(self, data_ingestion_artifact:Data_Ingestion_Artifact):
        pass"""
        
    def start_data_transformation(self, data_ingestion_artifact:Data_Ingestion_Artifact)->Data_Transformation_Artifact:
        try:
            logging.info(f"Starting the Data Transformation")
            data_transformation = Data_Transformation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact: Data_Transformation_Artifact = data_transformation.initiate_data_transforamtion()
        
            logging.info(f"Data Transformation Artifact Created!")
            return data_transformation_artifact
        
        except Exception as e:
            raise CustomException(e) from None

    def start_model_trainer(self, data_transformation_artifact:Data_Transformation_Artifact)->Model_Trainer_Artifact:
        try:
            logging.info(f"Starting the Model Trainer")
            model_trainer = Model_Trainer(data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact: Model_Trainer_Artifact = model_trainer.initiate_model_trainer()
        
            logging.info(f"Model Trainer Artifact Created!")
            return model_trainer_artifact
        
        except Exception as e:
            raise CustomException(e) from None
       
    """def start_model_evaluation(self, model_trainer_artifact:Model_Trainer_Artifact):
        pass
    
    def start_model_pusher(self, model_evaluation_artifact):
        pass"""
        
        
    def initiate_training_pipeline(self):
        try:
            # Execute the Training Pipeline Steps
            data_ingestion_artifact = self.start_data_ingestion()
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact)
            mode_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
        
            # temperorary
            return mode_trainer_artifact
        
        except Exception as e:
            raise CustomException(e) from None


    