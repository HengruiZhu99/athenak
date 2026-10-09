#!/usr/bin/env python3
"""Hash/finite-JSON checks and optional exact hex-input replay; no NPZ/eigensolver."""
from pathlib import Path
import argparse,hashlib,json
import dyadic_certificate as dc
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
def decode(rows):return [[complex(float.fromhex(z[0]),float.fromhex(z[1])) for z in row] for row in rows]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--metadata-only',action='store_true');a=p.parse_args()
    index=read(a.root/'index.json');checked=0;omitted=[]
    for item in index['files']:
        f=a.root/item['path']
        if not f.exists():
            if a.metadata_only and item['role']=='large_payload':omitted.append(item['path']);continue
            raise RuntimeError('missing indexed file '+item['path'])
        assert sha(f)==item['sha256'] and f.stat().st_size==item['bytes'];checked+=1
        if f.suffix=='.json':read(f)
    replay=[]
    for N in (8,12,16):
        folder=a.root/f'N{N}-certificate001';receipt=read(folder/'receipt.json');certificate=read(folder/'certificate.json')
        assert receipt['error'] is None and receipt['sources_unchanged'] is True and receipt['certificate_computation_completed'] is True
        assert certificate['certified_positive_eigenvalues_at_least']==2
        exact=folder/'exact-binary64-input.json'
        if not exact.exists():
            assert a.metadata_only;replay.append({'N':N,'exact_replay_skipped':'large hex input absent under compact policy'});continue
        data=read(exact);result=dc.certify(decode(data['J']),decode(data['V']),decode(data['W']),decode([data['saved_lambda']])[0])
        assert result==certificate;replay.append({'N':N,'exact_replay_identical':True,'positive_count':result['certified_positive_eigenvalues_at_least']})
    print(json.dumps({'index_sha256':sha(a.root/'index.json'),'checked_files':checked,'omitted_large_payload':omitted,
       'metadata_only':a.metadata_only,'exact_replays':replay,'passed':True},indent=2,allow_nan=False))
if __name__=='__main__':main()
