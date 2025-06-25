from src.components.data_validation import DataValidation
from src.config.configurations import ConfigurationManager
from src.logging import logger

class DataValidationPipeline:
    def __init__(self):
        pass

    def start_data_validation(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config= data_validation_config)
        data_validation.start()

if __name__ == '__main__':
    obj = DataValidationPipeline()
    obj.start_data_validation()
