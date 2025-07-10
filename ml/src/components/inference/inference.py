from src.model_architecture import CustomDataset , LSTMAutoEncoder
from src.logging import logger
import torch
import pandas as pd
from src.utils.common import load_binary, read_yaml
from src.constant import PARAMS_FILE_PATH
from pathlib import Path
import os
import requests
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from datetime import datetime, timedelta


# --- Configuration ---
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL","http://localhost:9091")
STEP = 30
SEQUENCE_LENGTH = 20
METRICS = {
    "cpu_usage": '100 - avg by (instance) (rate(windows_cpu_time_total{mode="idle"}[1m])) * 100',
    "memory_available": 'windows_os_physical_memory_free_bytes',
    "disk_read" : 'rate(windows_logical_disk_read_bytes_total[1m])',
    "disk_write": 'rate(windows_logical_disk_write_bytes_total[1m])',
    "network_receive": 'rate(windows_net_bytes_received_total{nic="TP-Link Wireless USB Adapter"}[1m])',
    "network_transmit": 'rate(windows_net_bytes_sent_total{nic="TP-Link Wireless USB Adapter"}[1m])'
}

params = read_yaml(PARAMS_FILE_PATH)

SCALER_PATH = Path(r'artifacts/data_transformation/object/preprocessor.pkl')
MODEL_PARA = Path(r'artifacts/model_trainer/model_parameters')
scaler = load_binary(SCALER_PATH)



class Inferencing:
    def __init__(self):
        self.threshold = params.threshold.value
        self.scaler = scaler

    def query_latest_metrics(self):
        end = datetime.now()
        start = end - timedelta(seconds=45)  # Buffer window for rate-based metrics
        content = {}

        for metric_name, query in METRICS.items():
            url = f"{PROMETHEUS_URL}/api/v1/query_range" #PROMETHEUS_URL = "http://localhost:9090"
            params = {
                "query": query,
                "start": int(start.timestamp()),
                "end": int(end.timestamp()),
                "step": 30
            }
            response = requests.get(url, params=params).json()

            if response['status'] != 'success':
                print(f"Prometheus query failed for {metric_name}: {response.get('error', 'Unknown error')}")
                return pd.DataFrame()
            values = response['data']['result']

            if not values:
                print(f"No data for metric: {metric_name}")
                return pd.DataFrame()

            series = values[0]["values"]
            timestamps, metrics = zip(*series)
            content['timestamp'] = pd.to_datetime(timestamps, unit='s')
            content[metric_name] = list(map(float, metrics))
            df = pd.DataFrame(content)
        return df
           

    def inference(self, data):
        norm_df = self.scaler.transform(data.drop(['timestamp'], axis=1))
        dataset = CustomDataset(data=norm_df, seq=SEQUENCE_LENGTH)
        model = LSTMAutoEncoder()
        model.load_state_dict(torch.load(MODEL_PARA, map_location=torch.device('cpu')))
        model.eval()
        with torch.no_grad():
            for x, _ in dataset:
                x = x.unsqueeze(0).float()
                outputs = model(x)
                error = torch.mean((outputs - x)**2 ).item()
        return error
 
    def push_score_to_gateway(self, score):
        registry = CollectorRegistry()
        # Push score
        score_g = Gauge('anomaly_score', 'Raw anomaly score', ['instance'], registry=registry)
        score_g.labels(instance="inference-agent").set(score)

        # Push binary anomaly flag
        flag_g = Gauge('is_anomaly', 'Prdeicted anomaly', ['instance'], registry=registry)
        flag_g.labels(instance="inference-agent").set(1 if score > self.threshold else 0)

        push_to_gateway(PUSHGATEWAY_URL, job='anomaly_detector', registry=registry)
