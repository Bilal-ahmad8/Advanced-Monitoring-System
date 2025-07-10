from src.components.model_trainer import ModelTrainer
from src.config.configurations import ConfigurationManager

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def start_model_training(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(model_trainer_config)
        model_trainer.start()


if __name__ == '__main__':
    obj = ModelTrainerPipeline()
    obj.start_model_training()