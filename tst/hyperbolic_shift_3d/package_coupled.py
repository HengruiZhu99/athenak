"""Portable evidence with hashes; omit unrelated host process command lines."""
import argparse,hashlib,io,json,tarfile
from pathlib import Path

def digest(data):return hashlib.sha256(data).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    index=dict(format=1,archives=[])
    for run in a.runs:
        name=run.parent.name+'-'+run.name;target=a.output/(name+'.tar.gz')
        assert not target.exists(),target
        entry=dict(name=name,files={},source=str(run),redactions=[])
        with tarfile.open(target,'w:gz') as archive:
            for path in sorted(run.iterdir()):
                if not path.is_file():continue
                if path.name=='hyperbolic_shift_3d_coupled':continue # architecture-dependent; source retained
                raw=path.read_bytes();data=raw
                if path.name=='manifest.json':
                    m=json.loads(raw)
                    if isinstance(m.get('process_inventory'),str):
                        del m['process_inventory']
                        m['process_inventory_note']='Unrelated host process command lines omitted from this public evidence copy.'
                        entry['redactions'].append(dict(file=path.name,field='process_inventory',local_original_sha256=digest(raw)))
                    data=(json.dumps(m,indent=2)+'\n').encode()
                item=tarfile.TarInfo(path.name);item.size=len(data);item.mtime=0
                archive.addfile(item,io.BytesIO(data));entry['files'][path.name]=digest(data)
        entry.update(archive=target.name,sha256=digest(target.read_bytes()),bytes=target.stat().st_size)
        assert entry['bytes']<95_000_000,'Split this archive before pushing'
        index['archives'].append(entry)
    (a.output/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print([(x['name'],x['bytes']) for x in index['archives']])
if __name__=='__main__':main()
