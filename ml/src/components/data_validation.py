from src.logging import logger
from src.entity.config_entity import DataValidationConfig
import pandas as pd

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def start(self):
        TRAIN_VALIDATION_STATUS = True
        VAL_VALIDATION_STATUS = True

        train_data = pd.read_csv(self.config.source_train_path)
        val_data = pd.read_csv(self.config.source_val_path)

        data_schema = self.config.schema

        train_schema = dict(train_data.dtypes)
        val_schema = dict(val_data.dtypes)

        for key in data_schema:
            if train_schema:
                if key not in train_schema.keys() or data_schema[key] != train_schema[key]:
                    TRAIN_VALIDATION_STATUS = False
            if val_schema:
                if key not in val_schema.keys() or data_schema[key] != val_schema[key]:
                    VAL_VALIDATION_STATUS = False

        with open(self.config.status, 'w') as f:
            f.write(f'Train Dataset status: {TRAIN_VALIDATION_STATUS} \nValidation Dataset status: {VAL_VALIDATION_STATUS}')


        logger.info('Data Validation completed Check Status.txt for status!')

            

