from src.model_architecture import CustomDataset , LSTMAutoEncoder
from src.logging import logger
import torch
from torch.utils.data import DataLoader
import pandas as pd
from src.utils.common import load_binary, read_yaml
from src.constant import SCHEMA_FILE_PATH, PARAMS_FILE_PATH
from pathlib import Path

schema = read_yaml(SCHEMA_FILE_PATH)
params = read_yaml(PARAMS_FILE_PATH)

SCALER_PATH = Path(r'artifacts\data_transformation\object\preprocessor.pkl')
MODEL_PARA = Path(r'artifacts\model_trainer\model_parameters')
scaler = load_binary(SCALER_PATH)

class Inferencing:
    def __init__(self, data:Path, output_directory:Path):
        self.data_path = data
        self.output_directory = output_directory
        self.target_col = schema.TARGET_COLUMNS.keys()
        self.threshold = params.threshold
        self.scaler = scaler
        self.model = LSTMAutoEncoder()

    def inference(self, model, data):
        all_error = []
        model.eval()
        with torch.no_grad():
            for x, y in data:
                outputs = model(x)
                error = torch.mean((outputs - y)**2 , dim=(1,2))
                all_error.append(error)
        return torch.cat(all_error)

    def start(self):
        df = pd.read_csv(self.data_path)
        df = df[self.target_col]
        timestamp = df['timestamp'].values

        norm_df = self.scaler.transform(df.drop(['timestamp'], axis=1))

        dataset = CustomDataset(data=norm_df, seq=20)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        model = LSTMAutoEncoder()
        model.load_state_dict(torch.load(MODEL_PARA, map_location=torch.device('cpu')))

        mse_per_seq = self.inference(model, dataloader)
        seq_timestamp = timestamp[19:]

        pred_df = pd.DataFrame({'timestamp': seq_timestamp, 'error': mse_per_seq.numpy()})

        pred_df['anomaly'] = (pred_df['error'] > self.threshold.value).astype(int)

        pred_df.to_csv(self.output_directory, index=False)

        logger.info(f'Data inferencing completed and new data created at {self.output_directory}')




