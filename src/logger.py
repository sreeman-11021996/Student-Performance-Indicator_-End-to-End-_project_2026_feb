import logging
import os
from datetime import datetime

LOG_DIR_NAME = "logs"
LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y-%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(),LOG_DIR_NAME)
os.makedirs(logs_path,exist_ok=True) 

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format = "[ %(asctime)s ] - line:%(lineno)d - %(name)s - %(levelname)s - %(message)s"
)


    