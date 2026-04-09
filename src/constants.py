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
PREDICTION_DIR = 'predictions'

# ........................ Data Ingestion ........................
# -------- Config --------

DATA_INGESTION_DIR = 'data_ingestion'
RAW_DATA_FILENAME = 'raw_data'
TRAIN_DATA_FiILENAME = 'train'
TEST_DATA_FILENAME = 'test'

DATA_FILE_PATH = 'notebook\data\StudentsPerformance.csv'
TEST_SIZE = 0.2

# ........................ Data Transformation ........................
# -------- Config --------
DATA_TRANSFORMATION_DIR = 'data_transformation'
PREPROCESSED_OBJ_FILENAME = 'preprocessed.pkl'
TRANSFORMED_TRAIN_DATA_FiILENAME = 'train_transformed.npz'
TRANSFORMED_TEST_DATA_FiILENAME = 'test_transformed.npz'

TARGET_COLUMN = 'math score'


# ........................ Model Trainer ........................
# -------- Config --------
MODEL_TRAINER_DIR = 'model_trainer'
TRAINED_MODEL_FILENAME = 'model.pkl'

# -------- Artifact --------
MODELS = 'models'
CVAL_R2_SCORE = 'cval_r2_score'
OVERFIT_GAP = 'overfit_gap'
CVAL_R2_SCORE_STD = 'cval_r2_score_std'
TEST_R2_SCORE = 'test_r2_score'

# -------- Component --------
BASE_R2_SCORE = 0.6
BASE_OVERFIT_GAP = 0.1



# ........................ Prediction ........................
# -------- Config --------
PREDICTION_DIR = 'prediction'
PREDICTION_TIME_STAMP = '2026-xx-xx__xx-xx-xx'
PREPROCESSED_OBJ_FILENAME = 'preprocessed.pkl'
MODEL_FILENAME = 'model.pkl'

PREDICTION_FILE_NAME = 'y_prediction.npz'   # predictions/xxxx-xx-xx__xx-xx-xx/y_prediction.npz






