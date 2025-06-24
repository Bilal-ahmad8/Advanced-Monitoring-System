from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import DataIngestionConfig
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
