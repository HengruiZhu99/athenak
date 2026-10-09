"""Independent manual-struct readback of stopped N24 wave-map failure; never acceptance.

No original restart-reader import, kernel, native executable or PDE spectrum.
The parser is a byte-exact copy of the separately validated root preflight
parser. All samples, including field-guard breaches, remain in the report.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

ROOT=Path('/Users/hz0693/research/hyperboloidal')
HERE=Path(__file__).resolve().parent
PARSER=HERE/'independent_parser.py'
ABI=ROOT/'build-layer-research/time-projection-controls/rst-reader-gate/abi.json'
NAMES=['wave-map-N24-large-t2']
POINTS={'wave-map-N24-large-t2':[.1375,-.9625,-.2291666666666667]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p):return json.loads(Path(p).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
def main():
    np.seterr(all='raise')
    assert sha(PARSER)=='d743dfd004dd09dbe698889e585c0cc37e47afde019bfd79e2f2a542061c00fe'
    name=sys.argv[1];assert name in NAMES
    out=HERE/'attempts'/(name+'-001');out.mkdir(parents=True,exist_ok=False)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    case=ROOT/'build-layer-research/wave-map-native-t2-root-20261009/batch001'/name
    lp=case/'launch-receipt.json';launch=load(lp)
    assert launch['returncode']==-6 and not launch['passed_native_process_and_provenance']
    assert launch['sources_before_after_equal']
    assert (case/'protected-inputs-before.json').read_bytes()==(case/'protected-inputs-after.json').read_bytes()
    directory=case/'output'
    assert {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}==set(launch['outputs'])
    pins={str(p):sha(p) for p in [Path(__file__).resolve(),PARSER,ABI,lp,Path(launch['input_path']),case/'native.stdout',case/'native.stderr',case/'protected-inputs-before.json',case/'protected-inputs-after.json']}
    for rel,m in launch['outputs'].items():
        p=directory/rel;assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256'];pins[str(p)]=m['sha256']
    for rel,digest in load(ABI)['source_sha256'].items():
        assert sha(ROOT/rel)==digest;pins[str(ROOT/rel)]=digest
    dump(out/'pins-before.json',pins)
    module_spec=importlib.util.spec_from_file_location('independent_manual_struct_parser',PARSER)
    parser=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(parser)
    history_path,=directory.glob('*.z4c.user.hst')
    history=np.atleast_2d(np.loadtxt(history_path));assert history.shape[1]==15 and np.isfinite(history).all()
    assert (history[:,1]>0).all() and (np.diff(history[:,0])>0).all()
    n=24;h=2.2/n;c=-1.1+(0.5-3)*h+np.arange(n+6)*h
    zz,yy,xx=np.meshgrid(c,c,c,indexing='ij');mask=xx*xx+yy*yy+zz*zz<1
    xyz=np.stack([xx[mask],yy[mask],zz[mask]],axis=1);radius=np.sqrt(np.sum(xyz*xyz,axis=1))
    dist=np.max(np.abs(xyz-np.asarray(POINTS[name])),axis=1);point=int(np.argmin(dist));assert dist[point]<1e-12
    rows=[];previous=-1.;initial=None
    for p in sorted(directory.rglob('*.rst')):
        row={'path':str(p),'sha256':sha(p),'partial_diagnostic_only':True,'accepted_native_run':False}
        try:
            t,dt,cycle,q=parser.read(p,n);assert t>previous;previous=t
            row.update(time=t,dt=dt,cycle=cycle)
            active=q[:,mask];finite=np.isfinite(active)
            row['nonfinite_count25']=np.count_nonzero(~finite,axis=1).tolist()
            row['finite_extrema25']=[{'min':float(v[np.isfinite(v)].min()) if np.isfinite(v).any() else None,'max':float(v[np.isfinite(v)].max()) if np.isfinite(v).any() else None} for v in active]
            assert finite.all(),'nonfinite saved active fields'
            if t==0:initial=active.copy()
            a,b,c0,d,e,f=(active[k] for k in [1,2,3,4,5,6])
            cof=[d*f-e*e,c0*e-b*f,b*e-c0*d,a*f-c0*c0,b*c0-a*e,a*d-b*b]
            det=a*cof[0]+b*cof[1]+c0*cof[2]
            positive=bool((active[[0,18]]>0).all());spd=bool((a>0).all() and (cof[5]>0).all() and (det>0).all())
            row.update(positive_lapse_chi=positive,SPD_leading_minors=spd,det_min=float(det.min()),det_error_max=float(np.max(np.abs(det-1))))
            for field,label in [(18,'alpha'),(0,'chi')]:
                mi=int(np.argmin(active[field]));ma=int(np.argmax(active[field]))
                row[label+'_min_location']=xyz[mi].tolist();row[label+'_max_location']=xyz[ma].tolist()
            row['native_failure_cell_full25']=active[:,point].tolist()
            row['native_failure_cell_xyz']=xyz[point].tolist()
            if spd:
                tr=(cof[0]*active[8]+2*cof[1]*active[9]+2*cof[2]*active[10]+cof[3]*active[11]+2*cof[4]*active[12]+cof[5]*active[13])/det
                row['trace_error_max']=float(np.max(np.abs(tr)))
                row['saved_field_guards_satisfied']=positive and row['det_error_max']<=1e-10 and row['trace_error_max']<=1e-10
            else:row['saved_field_guards_satisfied']=False
            match=np.flatnonzero(np.abs(history[:,0]-t)<=1e-12);assert len(match)==1
            hr=history[int(match[0])];row['original_native_history_H_Mcon_Zcon_Theta']=hr[2:6].tolist()
            row['det_history_difference']=abs(float(hr[6])-row['det_error_max'])
            if spd:row['trace_history_difference']=abs(float(hr[7])-row['trace_error_max'])
            if initial is not None:
                delta=active-initial;loc=np.argmax(np.abs(delta),axis=1)
                row['deviation_from_t0_max25']=np.max(np.abs(delta),axis=1).tolist()
                row['deviation_from_t0_max_locations25']=xyz[loc].tolist()
                row['radial_field_deviation_max25']=[{'rlo':lo,'rhi':hi,'max25':np.max(np.abs(delta[:,(radius>=lo)&(radius<hi)]),axis=1).tolist()} for lo,hi in [(0,.25),(.25,.5),(.5,.75),(.75,.9),(.9,.95),(.95,1)]]
        except Exception as exc:
            row.update(diagnostic_error=repr(exc),saved_field_guards_satisfied=False)
        rows.append(row)
    for p,digest in pins.items():assert sha(p)==digest,p
    dump(out/'pins-after.json',pins);dump(out/'rows.json',rows)
    receipt={'diagnostic_protocol_completed':True,'partial_diagnostic_only':True,'accepted_native_run':False,'native_returncode':-6,'original_target_time':2.,'case':name,'arrays':len(rows),'last_saved_time':max(row['time'] for row in rows if 'time' in row),'saved_guard_failures':sum(not row.get('saved_field_guards_satisfied',False) for row in rows),'before_after_equal':True,'source_sha256':sha(__file__),'parser_sha256':sha(PARSER),'rows_sha256':sha(out/'rows.json'),'pins_sha256':sha(out/'pins-before.json'),'numpy_version':np.__version__,'new_kernel_calls':0,'new_native_steps':0,'scope':'Independent manual struct parser and cofactor/minor fields only. H/M/Z/Theta copied from original native history; no independent differentiated constraint calculation. No acceptance or causal instability claim.'}
    dump(out/'receipt.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
