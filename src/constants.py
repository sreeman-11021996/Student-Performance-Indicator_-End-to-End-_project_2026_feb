import os,sys


ROOT_DIR = os.getcwd()
ARTIFACT_DIR = 'artifacts'


# -------- Config --------
# Data Ingestion Config
DATA_INGESTION_DIR = 'data_ingestion'
RAW_DATA_FILENAME = 'raw_data'
TRAIN_DATA_FiILENAME = 'train'
TEST_DATA_FILENAME = 'test'

DATA_FILE_PATH = 'notebook\data\StudentsPerformance.csv'
TEST_SIZE = 0.2


# -------- Config --------
# Data Transformation Config
DATA_TRANSFORMATION_DIR = 'data_transformation'
TRANSFORMED_TRAIN_DATA_FiILENAME = 'train_transformed'
TRANSFORMED_TEST_DATA_FiILENAME = 'test_transformed'

TARGET_COLUMN = 'math score'

