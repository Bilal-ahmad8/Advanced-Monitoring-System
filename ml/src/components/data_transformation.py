from src.logging import logger
from src.entity.config_entity import DataTransformationConfig
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.utils.common import save_binary

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def start(self):
        content = None
        check_status = self.config.check_status
        with open(check_status, 'r') as f:
            content = f.read()
        
        if content:
            co_lst = content.split('\n')
            status = all('True' in line for line in co_lst)
        
        if status == True:
            TARGET_COLUMNS = self.config.target_columns.keys()

            train_data = pd.read_csv(self.config.train_ingested_data, usecols=TARGET_COLUMNS)
            val_data = pd.read_csv(self.config.val_ingested_data, usecols= TARGET_COLUMNS)

            

            train_data = train_data.drop(['timestamp'], axis=1)
            val_data = val_data.drop(['timestamp'], axis=1)

            preprocessor = MinMaxScaler()
            preprocessor.fit(train_data)
            features_name = preprocessor.get_feature_names_out()
            train_data[features_name] = preprocessor.transform(train_data)
            val_data[features_name] = preprocessor.transform(val_data)

            train_data.to_csv(self.config.transformed_train_data, index=False)
            val_data.to_csv(self.config.transformed_val_data, index=False)

            save_binary(value=preprocessor, path=self.config.preprocessor_path)

            logger.info(f'Data Transformed and loaded successfully plus preprocessor Object saved at {self.config.preprocessor_path}')

        else:
            logger.info('Data status is not right, Transformation Terminated!')



