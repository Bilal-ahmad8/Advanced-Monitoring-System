from src.components.inference.inference import Inferencing
from pathlib import Path

DATA = Path(r'test\test_prometheus_metrics.csv')
OUT_PATH = Path(r'test\predicted.csv')

class InferencePipeline:
    def __init__(self):
        pass

    def start_inferencing(self, data:Path, data_out:Path):
        predictor = Inferencing(data=data, output_directory=data_out)
        predictor.start()

if __name__ == '__main__':
    obj = InferencePipeline()
    obj.start_inferencing(data=DATA, data_out=OUT_PATH)