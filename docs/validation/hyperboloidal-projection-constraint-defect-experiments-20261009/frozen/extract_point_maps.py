#!/usr/bin/env python3
"""Additive readback of saved source calls; never launches the kernel."""
from pathlib import Path
import argparse, hashlib, json, math, sys, time, warnings
import numpy as np

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, x): Path(p).write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt',type=Path)
    args=parser.parse_args();p=args.attempt.resolve()
    dest=p/'point-map-readback';dest.mkdir(exist_ok=False)
    np.seterr(all='raise');warnings.filterwarnings('error',category=RuntimeWarning)
    start=time.monotonic()
    receipt=json.loads((p/'receipt.json').read_text())
    controls=json.loads((p/'held-source-linearity.json').read_text())
    calls=json.loads((p/'calls.json').read_text())
    assert receipt['sources_unchanged'] and controls['max_row_scaled_l2']<=5e-11
    n=controls['shared_points'];count=controls['basis_query_rows']
    assert count==n*8*3
    xparts=[];yparts=[];sources={};consumed=0
    for i,call in enumerate(calls):
        if consumed==count:break
        assert call['command'][-1]=='--source-batch' and call['returncode']==0
        assert consumed+call['rows']<=count
        paths=[p/'calls'/('%06d.%s'%(i,s)) for s in ('input','stdout','stderr')]
        for path,key in zip(paths,('input_sha256','stdout_sha256','stderr_sha256')):
            assert sha(path)==call[key];sources[str(path)]=call[key]
        assert paths[2].stat().st_size==0
        x=np.loadtxt(paths[0],ndmin=2);y=np.loadtxt(paths[1],ndmin=2)
        assert x.shape==(call['rows'],10) and y.shape==(call['rows'],150)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        xparts.append(x);yparts.append(y);consumed+=call['rows']
    assert consumed==count
    x=np.concatenate(xparts).reshape(n,8,3,10)
    y=np.concatenate(yparts).reshape(n,8,3,150)
    points=x[:,0,0,4:7].copy()
    assert np.array_equal(x[:,:,:,:2],np.zeros((n,8,3,2)))
    assert np.array_equal(x[:,:,:,3],np.zeros((n,8,3)))
    for c in range(8):
        for d in range(3):
            assert np.array_equal(x[:,c,d,2],np.full(n,c))
            assert np.array_equal(x[:,c,d,4:7],points)
            assert np.array_equal(x[:,c,d,7:10],np.tile(np.eye(3)[d],(n,1)))
    assert len(set(map(tuple,points)))==n
    rhs=y[:,:,:,:22].copy();raw=y[:,:,:,33:55].copy()
    normals=y[:,:,:,146:150].copy()
    archive=dest/'raw22-point-maps.npz'
    np.savez_compressed(archive,points=points,rhs=rhs,raw_input=raw,algebraic_normals=normals)
    with np.load(archive,allow_pickle=False) as saved:
        for key,value in [('points',points),('rhs',rhs),('raw_input',raw),('algebraic_normals',normals)]:
            assert np.array_equal(saved[key],value) and np.isfinite(saved[key]).all()
    schema={
      'J':0,'m':0,'phase':0,'point_order':'first encounter in pinned driver center/h/stencil order; explicit points below',
      'points':points.tolist(),'channels':['alpha_L0','metric_trace_L0','P_L0','Theta_physical_L0',
          'beta_L1','Lambda_L1','metric_STF_L2','independent_A_L2'],
      'WJet_order':[[1,0,0],[0,1,0],[0,0,1]],
      'array_shapes':{'points':list(points.shape),'rhs':list(rhs.shape),'raw_input':list(raw.shape),
          'algebraic_normals':list(normals.shape)},
      'array_axes':'point,channel,WJet_basis,raw22_field; points has point,Cartesian_component',
      'raw22_order':['chi','g00','g01','g02','g11','g12','g22','P','A00','A01','A02','A11','A12','A22',
          'Lambda0','Lambda1','Lambda2','Theta_physical','alpha','beta0','beta1','beta2'],
      'contraction':'field[p,f]=sum_c,d map[p,c,d,f]*WJet[p,c,d]; no extra r^L',
      'source_width':150,'rhs_source_columns':[0,22],'raw_input_source_columns':[33,55],
      'normal_source_columns':[146,150],
      'scope':'Actual point-dependent physical seed/lift and source maps at saved samples only; no interpolation of the maps, no new kernel query, no spectrum or propagation.',
      'source_receipt':str(p/'receipt.json'),'source_receipt_sha256':sha(p/'receipt.json'),
      'linearity_receipt_sha256':sha(p/'held-source-linearity.json'),
      'input_source_pins':receipt['source_before'],'query_output_pins':sources,
      'npz':str(archive),'npz_sha256':sha(archive),'npz_bytes':archive.stat().st_size,
      'large_payload_metadata_only_in_Git':True}
    write(dest/'schema-and-ordering.json',schema)
    write(dest/'receipt.json',{'passed_saved_point_map_readback':True,'kernel_queries':0,
       'command':[sys.executable,*sys.argv],'source_sha256':sha(__file__),'seconds':time.monotonic()-start,
       'source_attempt_gate_passed':receipt['passed_projection_defect_readback_gate'],
       'map_linearity_gate_passed':True,'points':n,'query_rows':count,
       'npz_sha256':sha(archive),'schema_sha256':sha(dest/'schema-and-ordering.json')})
    print(json.dumps(json.loads((dest/'receipt.json').read_text()),indent=2))
if __name__=='__main__':main()
