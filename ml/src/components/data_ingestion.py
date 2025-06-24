from src.logging import logger
from src.entity.config_entity import DataIngestionConfig
from src.utils.common import create_directories
import pandas as pd
import os

class DataIngestion:
    def __init__(self, config : DataIngestionConfig):
        self.config = config
    
    def start(self):
        data = pd.read_csv(self.config.source)
        data.to_csv(self.config.local_data_file)

        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)

        split_index = data.index[0] + pd.Timedelta(hours=self.config.train_data_time_interval)
        train_data = data[data.index < split_index]
        validation_data = data[data.index >= split_index]

        train_data.to_csv(self.config.train_ingested_data)
        validation_data.to_csv(self.config.val_ingested_data)

        logger.info('Data Ingested and splited in Train and Validation set!')
