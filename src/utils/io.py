import json
import os
from typing import Any, Dict, List, Union

def load_json(path: str) -> Any:
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data: Any, path: str, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def list_files(directory: str, extensions: List[str] = None) -> List[str]:
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if extensions:
                if any(filename.endswith(ext) for ext in extensions):
                    files.append(os.path.join(root, filename))
            else:
                files.append(os.path.join(root, filename))
    return sorted(files)
