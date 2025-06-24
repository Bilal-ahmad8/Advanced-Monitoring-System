from src.logging import logger
from src.utils.common import read_yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source: Path
    local_data_file: Path
    train_data_time_interval: float
    train_ingested_data: Path
    val_ingested_data : Path