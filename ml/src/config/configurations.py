from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (DataIngestionConfig, DataValidationConfig, DataTransformationConfig,
                                       ModelTrainerConfig, ModelEvaluationConfig)
from src.constant import *


class ConfigurationManager:
    def __init__(self, config=CONFIG_FILE_PATH, schema=SCHEMA_FILE_PATH, params = PARAMS_FILE_PATH):
        self.config = read_yaml(config)
        self.schema = read_yaml(schema)
        self.params = read_yaml(params)

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
    

    def get_model_trainer_config(self):
        config = self.config.model_trainer
        params = self.params.model_params
        threshhold_value = self.params.threshhold.value

        create_directories([config.root_dir])
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            model_directory=config.model_directory,
            model_params=params,
            anomaly_threshhold=threshhold_value,
            training_data=config.training_data,
            validation_data=config.validation_data
        )
        return model_trainer_config
    
    def get_model_evaluation_config(self):
        config = self.config.model_evaluation
        params = self.params.model_params
        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir= config.root_dir,
            model_parameters=config.model_parameters,
            model_params=params,
            metric= config.metric,
            validation_data= config.validation_data
        )
        return model_evaluation_config
