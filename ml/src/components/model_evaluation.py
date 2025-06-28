from src.logging import logger
from src.entity.config_entity import ModelEvaluationConfig
from src.model_architecture import LSTMAutoEncoder, CustomDataset
from src.utils.common import save_json
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import pandas as pd
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config:ModelEvaluationConfig):
        self.config = config

    def val(self, model, val_loader, criterion):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data , target in val_loader:
                output = model(data)
                loss = criterion(target, output)
                total_loss += loss.item()
            return total_loss/len(val_loader)

    def start(self):
        params = self.config.model_params
        val_data = pd.read_csv(self.config.validation_data)
        dataset = CustomDataset(val_data.values, 20)
        valloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False)

        model = LSTMAutoEncoder()
        model.load_state_dict(torch.load(self.config.model_parameters, map_location=torch.device('cpu')))
        logger.info('Model Parameters Loaded Succesfully!')

        criterion = nn.MSELoss()
        loss = self.val(model, valloader, criterion)

        save_json(Path(self.config.metric), {'MSELoss on Validation set': loss})

        logger.info(f'Model Evaluated check score at {self.config.metric}')

        