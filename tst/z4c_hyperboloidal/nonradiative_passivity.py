#!/usr/bin/env python3
"""Two-step deterministic CPU/MPI recorder comparison with complete receipts."""
import argparse,hashlib,json,sys
from pathlib import Path
from restart_roundtrip import run,compare
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'vis/python'))
from scri_reader import read

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('binary',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    binary=a.binary.resolve();out=a.output.resolve();out.mkdir(exist_ok=False)
    base=(Path(__file__).with_name('n12_initialize.athinput')).read_text()
    base+='\n<time>\nevolution=dynamic\nintegrator=rk4_classical\nnlim=2\ntlim=1\nndiag=1\n<output9>\nfile_type=rst\ndt=100\n'
    receipt={'executable':str(binary),'executable_sha256':sha(binary),'cases':[]};checkpoints=[]
    for name in ('off','on'):
        text=base+('\n<output1>\nfile_type=scri\ndcycle=1\nboundary_diagnostics=true\n' if name=='on' else '')
        inp=out/(name+'.athinput');inp.write_text(text)
        checkpoint=run(binary,out,name,2,['-i',str(inp)]);checkpoints.append(checkpoint)
        receipt['cases'].append({'name':name,'input_sha256':sha(inp),'checkpoint':checkpoint['path'],
            'checkpoint_sha256':sha(Path(checkpoint['path'])),'command':['mpiexec','-n','2',str(binary),'-i',str(inp)],'cwd':str(out/name),'OMP_NUM_THREADS':1})
    receipt['comparison']=compare(*checkpoints)
    epoch=read(next((out/'on').glob('*.epoch.scri')))
    (out/'recorder-source-manifest.txt').write_text(epoch['source_manifest'])
    receipt['recorder_manifest_sha256']=sha(out/'recorder-source-manifest.txt')
    assert sha(binary)==receipt['executable_sha256']
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
