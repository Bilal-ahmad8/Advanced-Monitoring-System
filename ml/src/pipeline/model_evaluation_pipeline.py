from src.components.model_evaluation import ModelEvaluation
from src.config.configurations import ConfigurationManager

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def start_model_evaluation(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        evaluate_model = ModelEvaluation(model_evaluation_config)
        evaluate_model.start()

if __name__ == '__main__':
    obj = ModelEvaluationPipeline()
    obj.start_model_evaluation()