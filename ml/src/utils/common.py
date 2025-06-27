import os, yaml, json
from pathlib import Path
from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError
import pickle

@ensure_annotations
def read_yaml(path: Path) -> ConfigBox:
    try:
        with open(path) as file:
            content = yaml.safe_load(file)
        return ConfigBox(content)
    except BoxValueError:
        raise ValueError('Yaml File is Empty')
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path:list):
    for p in path:
        os.makedirs(p, exist_ok=True)


def save_binary(value, path):
    with open(path, 'wb') as f:
        pickle.dump(value,f)

@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        