from src.pipelines.train_pipeline import Training_Pipeline 



if __name__ == "__main__":
      
      train_pipeline = Training_Pipeline()
      model_trainer_artifact = train_pipeline.initiate_training_pipeline()
    
      print("Model Trainer Completed \n"
            f"Trained Model Name : {model_trainer_artifact.trained_model_name}\n"
            f"Trained Model File Path : {model_trainer_artifact.trained_model_file_path}\n"
            f"Trained model metrics : {model_trainer_artifact.trained_model_metrics}\n\n")


      
      
      # 4. Model Factory

      # ** CHange src.constants.py for production!!!!
    