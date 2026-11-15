#!/usr/bin/env python3
"""Generate pulse seeds on Aurora; retain the exact native restart identity."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np

root=Path(sys.argv[1]).resolve()
assert root==Path('/lus/flare/projects/CompactBinaryMerger/hzhu/scri-regularity-20261110')
sys.path.insert(0,str(root/'analysis-v1'))
from restart_roundtrip import parse,payload,compare
out=root/'pulse-seeds-gpu-v1';assert out.is_dir()
assert not (out/'manifest.json').exists()
cpu=root/'pulse-seeds-v1'
planned=json.loads((cpu/'manifest.json').read_text());assert len(planned)==14
records=[];states={}
mpi=['mpiexec','-n','24','-ppn','12','--cpu-bind=list:'+os.environ['PULSE_MPI_BINDING'],
     os.environ['PULSE_GPU_WRAPPER']]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
for item in planned:
    case=item['case'];d=out/case;d.mkdir()
    original=cpu/case/'input.athinput'
    assert digest(original)==item['input_sha256']
    # Native diagnostics always use one component here; make the legacy hint
    # consistent as well. No evolved field or formulation parameter is changed.
    text=original.read_text().replace('num_horizons = 0','num_horizons = 1')
    inp=d/'input.athinput';inp.write_text(text)
    command=mpi+[str(root/'pulse-writer-v1/scri_pulse'),str(inp),str(item['amplitude'])]
    print('Generating',case,flush=True)
    with (d/'seed.log').open('w') as log:
        subprocess.run(command,cwd=str(d),stdout=log,stderr=subprocess.STDOUT,timeout=900,check=True)
    paths=list((d/'rst').glob('*.rst'));assert len(paths)==1
    info=parse(paths[0]);assert info['time']==0 and info['cycle']==0
    states[case]=info
    record={k:item[k] for k in ['case','n','mode','amplitude']}
    record.update(checkpoint=str(paths[0].relative_to(out)),checkpoint_sha256=digest(paths[0]),
                  input_sha256=digest(inp),cpu_input_sha256=item['input_sha256'],command=command,
                  end_M0=3*info['dt']*(1422*item['n']//12))
    records.append(record)
    (out/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')

validation={}
for mode,job in [('A','9064725'),('B','9065387')]:
    path=root/'runs'/('{}-N12-{}'.format(mode,job))/'rst'/('scri_{}-N12.00000.rst'.format(mode))
    assert path.exists(),path
    validation[mode+'_zero_equals_production']=compare(states[mode+'-N12-control'],parse(path))
    base,full,half=[payload(states[mode+'-N12-'+name]) for name in ['control','pulse','halfamp']]
    for f in range(39):
        if f not in (13,18):
            assert np.array_equal(base[:,:,f],full[:,:,f])
            assert np.array_equal(base[:,:,f],half[:,:,f])
    error=0.
    for f in (13,18):
        error=max(error,float(np.max(abs((full[:,:,f]-base[:,:,f])-2*(half[:,:,f]-base[:,:,f])))))
    assert error<5e-15
    validation[mode+'_amplitude_linearity_error']=error
    validation[mode+'_other_fields_unchanged']=True
validation['writer_executable_sha256']=digest(root/'pulse-writer-v1/scri_pulse')
validation['writer_source_sha256']=digest(root/'pulse-writer-v1/source.cpp')
validation['production_executable_sha256']=digest(root/'build-v1/src/athena')
(out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(validation,indent=2),flush=True)
