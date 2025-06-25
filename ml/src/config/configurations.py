from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (DataIngestionConfig, DataValidationConfig, DataTransformationConfig)
from src.constant import *


class ConfigurationManager:
    def __init__(self, config=CONFIG_FILE_PATH, schema=SCHEMA_FILE_PATH):
        self.config = read_yaml(config)
        self.schema = read_yaml(schema)
        #self.params = read_yaml(params)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self):
        config = self.config.data_ingestion
        create_directories([config.root_dir, config.feature_store, config.split_store])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source=config.source,
            local_data_file=config.local_data_file,
            train_data_time_interval= config.train_data_time_interval,
            train_ingested_data=config.train_ingested_data,
            val_ingested_data=config.val_ingested_data
        )
        return data_ingestion_config
    
    def get_data_validation_config(self):
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir= config.root_dir,
            source_train_path = config.source_train_path,
            source_val_path = config.source_val_path,
            schema= schema,
            status = config.status,
        )
        return data_validation_config
    
    def get_data_transformation_config(self):
        config = self.config.data_transformation
        schema = self.schema.TARGET_COLUMNS
        create_directories([config.root_dir, config.data_directory, config.object_directory])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            check_status=config.check_status,
            data_directory=config.data_directory,
            object_directory=config.object_directory,
            train_ingested_data=config.train_ingested_data,
            val_ingested_data=config.val_ingested_data,
            target_columns= schema,
            transformed_train_data=config.transformed_train_data,
            transformed_val_data=config.transformed_val_data,
            preprocessor_path=config.preprocessor_path
        )
        return data_transformation_config
