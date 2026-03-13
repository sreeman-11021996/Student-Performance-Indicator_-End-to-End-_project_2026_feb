from src.components.data_ingestion import Data_Ingestion




if __name__ == "__main__":
    obj = Data_Ingestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()

    print(f"Train Data Path : {train_data_path} \nTest Data Path : {test_data_path}")