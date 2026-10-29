"""Read-only verification of the portable coupled-evolution evidence archives."""
import argparse,hashlib,json,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
index=json.loads((a.directory/'index.json').read_text());total=0
for entry in index['archives']:
    path=a.directory/entry['archive'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],path
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers();assert {m.name for m in members}==set(entry['files'])
        for member in members:
            assert member.isfile() and '/' not in member.name and member.name not in ['.','..']
            data=archive.extractfile(member).read()
            assert hashlib.sha256(data).hexdigest()==entry['files'][member.name],(path,member.name)
            if member.name=='manifest.json':
                manifest=json.loads(data)
                assert manifest['status'] in ['completed','bounded_stop','failed','external_timeout','setup_or_run_error']
                if 'result' in manifest:assert manifest['status']==manifest['result']['status']
            total+=1
print(json.dumps(dict(archives=len(index['archives']),verified_files=total)))
