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


if __name__=='__main__':
    import hashlib
    import sys
    state=json.loads((trajectory_directory(Path(sys.argv[1]),sys.argv[2])/'state.json').read_text())
    executable=Path(state['executable'])
    with executable.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=state['binary_sha256']:
            raise ValueError('accepted N12 executable digest changed')
    print(executable)
