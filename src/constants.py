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



# ........................ Model Factory ........................

# Model Factory files
MODEL_CONFIG_FILENAME = 'model.yaml'

# Model Factory keys
GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"
GRID_SEARCH_PARAMS_KEY = 'grid_search_params'

# Untuned Model class
MODEL_NAME_KEY = 'model_name'
MODEL_NUMBER_KEY = 'model_serial_number'

# Grid Search CV 
GRID_SEARCH_RESULT_LIST_KEY = 'grid_search_result_list'
MODEL_KEY = 'model'
MEAN_TEST_R2_KEY = 'mean_test_score'
MEAN_TRAIN_R2_KEY = 'mean_train_score'
STD_TEST_R2_KEY = 'std_test_score'

# Grid Searched Model class
VAL_R2_KEY = 'val_r2_score'
VAL_R2_STD_KEY = 'val_r2_std'
OVERFIT_GAP_KEY = 'overfit_gap'

# Best Model class
BASE_R2 = 0.6
OVERFIT_GAP = 0.1



# ........................ Prediction ........................
# -------- Config --------
PREDICTION_DIR = 'prediction'
PREDICTION_TIME_STAMP = '2026-xx-xx__xx-xx-xx'
PREPROCESSED_OBJ_FILENAME = 'preprocessed.pkl'
MODEL_FILENAME = 'model.pkl'

PREDICTION_FILE_NAME = 'y_prediction.npz'   # predictions/xxxx-xx-xx__xx-xx-xx/y_prediction.npz






