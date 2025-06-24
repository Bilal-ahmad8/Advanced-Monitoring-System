from src.logging import logger
from src.components.data_ingestion import DataIngestion
from src.config.configurations import ConfigurationManager


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def start_data_ingestion(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.start()
        
if __name__ =="__main__":
    obj = DataIngestionTrainingPipeline()
    obj.start_data_ingestion()