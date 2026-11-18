"""Resolve isolated trajectory directories shared by evolution and endpoint analysis."""
import json
from pathlib import Path


def trajectory_directory(root, mode):
    base = root/'n12-10M'
    selection = base/'multirate-trial.json'
    if mode=='multirate' and selection.is_file():
        name = json.loads(selection.read_text())['directory']
        if Path(name).name!=name or not name.startswith('multirate-'):
            raise ValueError('invalid isolated multirate trial directory')
        return base/name
    return base/mode
