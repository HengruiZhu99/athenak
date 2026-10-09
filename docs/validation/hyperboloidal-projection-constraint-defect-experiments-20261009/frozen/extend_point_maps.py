#!/usr/bin/env python3
"""Combine original and refined saved point maps; no kernel queries."""
from pathlib import Path
import argparse, hashlib, json, sys, time, warnings
import numpy as np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('attempt',type=Path)
    a=parser.parse_args();p=a.attempt.resolve();root=p.parents[1]
    old=root/'attempt-1791558113840144000/point-map-readback'
    base=old/'raw22-point-maps.npz';schema=old/'schema-and-ordering.json'
    assert sha(base)=='8f419ece7c5e6b0b6318adefd4fca621d200bfb3019410d14ead849ce086231c'
    assert sha(schema)=='53e58a192492bdf23a069e51b1b36415f246309156c3f9934c0394c5c6b0e779'
    np.seterr(all='raise');warnings.filterwarnings('error',category=RuntimeWarning)
    dest=p/'point-map-readback-union';dest.mkdir(exist_ok=False);start=time.monotonic()
    reuse=json.loads((p/'cache-reuse.json').read_text());linearity=json.loads((p/'held-source-linearity.json').read_text())
    receipt=json.loads((p/'receipt.json').read_text());assert receipt['sources_unchanged']
    assert linearity['max_row_scaled_l2']<=5e-11
    count=reuse['new_basis_query_rows'];n=reuse['new_points'];assert count==n*24
    xp=[];yp=[];consumed=0;pins={str(base):sha(base),str(schema):sha(schema)}
    for i,call in enumerate(json.loads((p/'calls.json').read_text())):
        if consumed==count:break
        assert call['command'][-1]=='--source-batch' and call['returncode']==0 and consumed+call['rows']<=count
        paths=[p/'calls'/('%06d.%s'%(i,s)) for s in ('input','stdout','stderr')]
        for path,key in zip(paths,('input_sha256','stdout_sha256','stderr_sha256')):
            assert sha(path)==call[key];pins[str(path)]=call[key]
        assert paths[2].stat().st_size==0
        x=np.loadtxt(paths[0],ndmin=2);y=np.loadtxt(paths[1],ndmin=2)
        assert x.shape==(call['rows'],10) and y.shape==(call['rows'],150)
        assert np.isfinite(x).all() and np.isfinite(y).all();xp.append(x);yp.append(y);consumed+=call['rows']
    assert consumed==count
    x=np.concatenate(xp).reshape(n,8,3,10);y=np.concatenate(yp).reshape(n,8,3,150)
    points=x[:,0,0,4:7].copy();assert np.array_equal(points,np.asarray(reuse['new_point_order']))
    assert np.array_equal(x[:,:,:,:2],np.zeros((n,8,3,2))) and np.array_equal(x[:,:,:,3],np.zeros((n,8,3)))
    for c in range(8):
        for d in range(3):
            assert np.array_equal(x[:,c,d,2],np.full(n,c)) and np.array_equal(x[:,c,d,4:7],points)
            assert np.array_equal(x[:,c,d,7:10],np.tile(np.eye(3)[d],(n,1)))
    with np.load(base,allow_pickle=False) as b:
        arrays={key:np.concatenate((b[key],value)) for key,value in [('points',points),('rhs',y[:,:,:,:22]),
            ('raw_input',y[:,:,:,33:55]),('algebraic_normals',y[:,:,:,146:150])]}
    assert len(set(map(tuple,arrays['points'])))==len(arrays['points'])
    archive=dest/'raw22-point-maps-union.npz';np.savez_compressed(archive,**arrays)
    with np.load(archive,allow_pickle=False) as check:
        for key,value in arrays.items():assert np.array_equal(check[key],value) and np.isfinite(check[key]).all()
    out=json.loads(schema.read_text());out.update({'point_order':'Original 4809 points, then 882 fresh half-h points; explicit points below',
        'points':arrays['points'].tolist(),'array_shapes':{k:list(v.shape) for k,v in arrays.items()},
        'union_sources':pins,'refined_attempt':str(p),'refined_receipt_sha256':sha(p/'receipt.json'),
        'refined_cache_reuse_sha256':sha(p/'cache-reuse.json'),'refined_linearity_sha256':sha(p/'held-source-linearity.json'),
        'npz':str(archive),'npz_sha256':sha(archive),'npz_bytes':archive.stat().st_size,
        'scope':'Exact saved actual point-map union only; both original/refined ordinary-FD gates remain failed. No new kernel query or interpolation.'})
    write(dest/'schema-and-ordering.json',out)
    write(dest/'receipt.json',{'passed_saved_map_union_readback':True,'kernel_queries':0,
        'command':[sys.executable,*sys.argv],'source_sha256':sha(__file__),'seconds':time.monotonic()-start,
        'union_points':len(arrays['points']),'new_points':n,'source_attempt_gate_passed':receipt['passed_projection_defect_readback_gate'],
        'map_linearity_gate_passed':True,'npz_sha256':sha(archive),'schema_sha256':sha(dest/'schema-and-ordering.json')})
    print((dest/'receipt.json').read_text())
if __name__=='__main__':main()
