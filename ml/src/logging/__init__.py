import logging
import os
from datetime import datetime

log_file = f"{ datetime.now().strftime('%m_%d_%Y_%H_%M_%S') }.log"
                                      
log_path = os.path.join(os.getcwd(), "logs", log_file)
os.makedirs(log_path, exist_ok=True)

log_file_path = os.path.join(log_path, log_file)

logger = logging.getLogger(logging.basicConfig(
    format= '[ %(asctime)s: ] %(lineno)d - %(name)s - %(levelname)s - %(message)s ',
    level = logging.INFO,
    filename= log_file_path
))