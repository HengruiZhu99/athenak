#!/usr/bin/env python3
"""Corruption controls for offline AMR readers, using untouched passing CPU data."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from native_amr_gpu_compare import (ROW, checkpoint_audit, read_snapshot, require, sha)
from restart_roundtrip import HEADER, parse


def rejected(call, expected):
    try:
        call()
    except (ValueError, AssertionError) as error:
        require(expected in str(error), 'unexpected control rejection: '+str(error))
        return dict(rejected=True, reason=str(error))
    raise AssertionError('corruption was accepted: '+expected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evolution-two',type=Path,required=True)
    parser.add_argument('--restart-two',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'new output directory required')
    args.output.mkdir(parents=True)
    source=args.evolution_two.resolve()
    baseline,_,_=read_snapshot(source,'cycle4',2,False,True,{})
    original=parse(args.restart_two.resolve())
    before={str(p):sha(p) for p in source.glob('cycle4-rank*')}
    before[str(args.restart_two.resolve())]=sha(args.restart_two)
    checks={}
    with tempfile.TemporaryDirectory(dir=args.output) as temporary:
        folder=Path(temporary)
        for path in source.glob('cycle4-rank*'):
            (folder/path.name).symlink_to(path)
        target=folder/'cycle4-rank1.bin'
        target.unlink()
        shutil.copyfile(source/target.name,target)
        data=np.memmap(target,dtype=ROW,mode='r+')
        surfaces=np.flatnonzero(data['keys'][:,5]==1)
        require(len(surfaces)>0,'control rank lacks surface point')
        q=int(surfaces[-1])
        old=float(data['values'][q,30])
        data['values'][q,30]=old+.125
        data.flush()
        mutated,_,_=read_snapshot(folder,'cycle4',2,False,True,{})
        checks['late_scri_Jz']=rejected(lambda:require(mutated==baseline,
            'cross-rank complete31+8 AMR state differs'),'complete31+8')
        data['values'][q,30]=old
        data['keys'][q]=data['keys'][q-1]
        data.flush()
        checks['duplicate_late_point']=rejected(lambda:read_snapshot(folder,'cycle4',2,False,True,{}),
                                               'missing/duplicate/extra')
        del data
        target.unlink()
        target.symlink_to(source/target.name)
        meta=folder/'cycle4-rank1.txt'
        meta.unlink()
        lines=(source/meta.name).read_text().splitlines()
        fields=lines[-1].split();fields[5]=str(int(fields[5])+1);lines[-1]=' '.join(fields)
        meta.write_text('\n'.join(lines)+'\n')
        checks['wrong_owned_bounds']=rejected(lambda:read_snapshot(folder,'cycle4',2,False,True,{}),
                                             'evolved bounds differ')
        bad=folder/'checkpoint.rst'
        shutil.copyfile(args.restart_two,bad)
        info=parse(bad)
        payload=np.memmap(bad,dtype='<f8',mode='r+',offset=info['payload_at'],
                          shape=(info['blocks'],18**3+144,39))
        active=np.argwhere(payload[:,:,31]!=0)
        m,p=map(int,active[-1]);old=payload[m,p,38]
        payload[m,p,38]=np.nan;payload.flush()
        checks['nonfinite_last_history_member']=rejected(lambda:checkpoint_audit(bad,2,72),
                                                       'wrong/nonfinite complete restart payload')
        payload[m,p,38]=old;payload.flush();del payload
        _,invariant,_=checkpoint_audit(bad,2,72)
        with bad.open('r+b') as stream:
            stream.seek(info['start']+120)
            stream.write(np.arange(9,dtype='<i4').tobytes())
        _,same,_=checkpoint_audit(bad,2,72)
        require(same==invariant,'unused coarse metadata changed meaningful identity')
        checks['unused_coarse_header_neutral']=dict(accepted=True, initialized_identity_unchanged=True)
        with bad.open('r+b') as stream:
            stream.seek(info['start']+HEADER-12)
            stream.write(np.array([info['dt']*2],dtype='<f8').tobytes())
        checks['wrong_dt']=rejected(lambda:checkpoint_audit(bad,2,72),'exact time changed')
    require(all(sha(Path(name))==digest for name,digest in before.items()),'CPU fixtures changed')
    result=dict(passed=True,scope='offline AMR reader corruption controls; no GPU execution',
        controls=checks,original_inputs_unchanged=True,input_sha256=before,
        source_sha256={name:sha(Path(__file__).with_name(name)) for name in
            ('native_amr_gpu_controls.py','native_amr_gpu_compare.py','amr_restart.py','restart_roundtrip.py')})
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,controls=len(checks)),indent=2))


if __name__=='__main__':
    main()
