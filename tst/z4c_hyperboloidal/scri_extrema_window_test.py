#!/usr/bin/env python3
"""Verify exact unions of stage extrema, including restart provenance guards."""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
from scri_compare_endpoints import extrema_window


def verify():
    rejected=[]
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);manifest={'streams':{}}
        for group in range(4):
            name=f'test.scri-investigation-{group}.hst'
            data=np.array([[0,.1,0,0,0],[1,.1,-7,2,8],[1.5,.1,-3,10,4],[2,.1,-1,3,6]])
            np.savetxt(root/name,data,header='[1]=time [2]=dt [3]=signed-min [4]=signed-max [5]=absolute')
            rows=[dict(time_native=t,window_start_native=start,complete_window=True)
                  for t,start in zip(data[:,0],[0,0,1,1.5])]
            manifest['streams'][name]={'rows':rows}
        p=root/'merge-manifest.json';p.write_text(json.dumps(manifest))
        signed,absolute,sources=extrema_window(root,0,6)
        assert signed['signed']=={'min':-7.,'max':10.,'t_M0':6.}
        assert absolute['absolute']=={'max':8.,'t_M0':6.}
        assert all(s['subwindows']==3 for s in sources.values())
        signed,absolute,_=extrema_window(root,3,6)
        assert signed['signed']['min']==-3 and absolute['absolute']['max']==6

        def reject(name,begin=0):
            try:extrema_window(root,begin,6)
            except AssertionError:rejected.append(name)
            else:raise AssertionError('accepted invalid window: '+name)

        reject('unrecorded_endpoint',begin=.1)
        row=manifest['streams']['test.scri-investigation-0.hst']['rows'][2]
        row['complete_window']=False;p.write_text(json.dumps(manifest));reject('lost_restart_window')
        row['complete_window']=True;row['window_start_native']=1.1
        p.write_text(json.dumps(manifest));reject('gap')
        row['window_start_native']=.9;p.write_text(json.dumps(manifest));reject('overlap')
    return dict(exact_signed_union=True,exact_absolute_union=True,rejected_invalid_windows=rejected)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=verify()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
