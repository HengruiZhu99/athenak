"""Compare all fields, reduction/curl components, and trackers across restart/MPI."""
import argparse
import json
from pathlib import Path
import numpy as np
from verify_pulses import table


def compare(reference, experiment):
    a,b=table(reference,'final'),table(experiment,'final')
    a=a[np.lexsort((a['z'],a['y'],a['x']))]
    b=b[np.lexsort((b['z'],b['y'],b['x']))]
    assert len(a)==len(b) and a.dtype.names==b.dtype.names
    for name in ['x','y','z','time','volume']:
        assert np.array_equal(a[name],b[name]),name
    errors={name:float(np.max(np.abs(a[name]-b[name]))) for name in a.dtype.names
            if name.startswith(('u','E','C'))}
    assert len([n for n in errors if n.startswith('u')])==55
    assert len([n for n in errors if n.startswith('E')])==33
    assert len([n for n in errors if n.startswith('C')])==33
    assert all(np.isfinite(v) for v in errors.values())
    trackers=[]
    for n in range(2):
        pa=list(reference.glob(f'*.co_{n}.txt'));pb=list(experiment.glob(f'*.co_{n}.txt'))
        assert len(pa)==len(pb)==1
        ta,tb=[np.atleast_2d(np.loadtxt(p)) for p in [pa[0],pb[0]]]
        delta=float(abs(ta[-1,1:]-tb[-1,1:]).max())
        trackers.append(delta)
    result=dict(reference=str(reference),experiment=str(experiment),
                max_component_difference=max(errors.values()),components=errors,
                tracker_final_differences=trackers,
                tolerance=2e-12,passed=max(list(errors.values())+trackers)<=2e-12)
    if not result['passed']: raise ValueError(result)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reference',type=Path);ap.add_argument('experiment',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();r=compare(a.reference,a.experiment)
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
