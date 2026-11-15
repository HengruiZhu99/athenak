#!/usr/bin/env python3
"""Exercise field-order inference and reject mismatched temporal-control evidence."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from scri_field_convergence import compare


def verify():
    rejected=[]
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        samples=[]; receipts=[]
        for name,n,dt in [('N12',12,.02),('N16',16,.015),('N20',20,.012),('half',12,.01)]:
            sample=root/(name+'.npz'); receipt=root/(name+'-receipt.json')
            values=np.arange(1,373,dtype=float).reshape(2,3,2,1,31)/n**4
            if name=='half':
                values+=1e-9
            np.savez(sample,fields_w6=values,fields_w8=values,
                     coordinates=np.zeros((2,3,2,1,3)),radii=np.array([.95,.98,1.]),angles=np.array([.25,.75]))
            digest=hashlib.sha256(name.encode()).hexdigest()
            metadata=dict(n=n,time_M0=20.,cycle=200 if name=='half' else 100,
                          checkpoint_sha256=digest,physics_identity_without_resolution_and_dt='native;mode=B')
            sample.with_suffix('.json').write_text(json.dumps(metadata))
            native=dict(sha256=digest,cycle=metadata['cycle'],time_M0=20.,dt_native=dt,
                        identity=f'native;n={n},{n},{n};p.requested_dt={dt.hex()};mode=B')
            receipt.write_text(json.dumps(native))
            samples.append(sample);receipts.append(receipt)
        args=(samples[:3],samples[3],receipts[:3],receipts[3])
        result=compare(*args)
        for region in result['regions'].values():
            for group in region.values():
                assert abs(group['effective_order']-4)<1e-10
                assert group['positive_aligned_order_above_sampling_sensitivity']
                assert abs(group['N12_half_dt_difference_rms']-1e-9)<1e-17

        def reject(name, call):
            try:
                call()
            except (AssertionError,ValueError):
                rejected.append(name)
            else:
                raise AssertionError('accepted invalid evidence: '+name)

        reject('missing_native_receipts',lambda:compare(samples[:3],samples[3]))
        reject('missing_halfdt_receipt',lambda:compare(samples[:3],samples[3],receipts[:3]))
        original=json.loads(receipts[3].read_text())
        bad_fields={'sha256':'wrong','cycle':999,'time_M0':19.,'dt_native':.02,
                    'identity':'native;n=12,12,12;p.requested_dt=0x1p-2;mode=A'}
        for field,value in bad_fields.items():
            receipts[3].write_text(json.dumps(dict(original,**{field:value})))
            reject('halfdt_'+field,lambda:compare(*args))
        receipts[3].write_text(json.dumps(original))
        original_main=json.loads(receipts[1].read_text())
        receipts[1].write_text(json.dumps(dict(original_main,dt_native=.02)))
        reject('main_sequence_dt_scaling',lambda:compare(*args))
        receipts[1].write_text(json.dumps(original_main))
        with np.load(samples[3]) as data:
            arrays={k:data[k] for k in data.files}
        arrays['angles']=np.array([.2,.8]);np.savez(samples[3],**arrays)
        reject('halfdt_sampling_angles',lambda:compare(*args))
    return dict(manufactured_order=4,known_temporal_difference=1e-9,rejected_invalid_evidence=rejected)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=verify()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
