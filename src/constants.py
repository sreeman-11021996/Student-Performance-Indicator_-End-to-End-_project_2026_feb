import os,sys
from datetime import datetime


# cwd/config/config.yaml
ROOT_DIR = os.getcwd() 
CONFIG_DIR = 'config'
CONFIG_FILE_NAME = 'config.yaml'
CONFIG_FILE_PATH = os.path.join(ROOT_DIR,CONFIG_DIR,CONFIG_FILE_NAME)

# time_stamp
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}"

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
PREPROCESSED_OBJ_FILENAME = 'preprocessed_obj.pkl'
TRANSFORMED_TRAIN_DATA_FiILENAME = 'train_transformed.npz'
TRANSFORMED_TEST_DATA_FiILENAME = 'test_transformed.npz'

TARGET_COLUMN = 'math score'

