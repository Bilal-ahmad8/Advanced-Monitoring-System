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

@dataclass
class DataValidationConfig:
    root_dir: Path
    source_train_path: Path
    source_val_path: Path
    schema : dict
    status: Path