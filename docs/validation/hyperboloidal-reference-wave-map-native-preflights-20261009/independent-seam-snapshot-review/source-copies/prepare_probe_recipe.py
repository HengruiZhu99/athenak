"""Prepare, but never execute, a fixed native-array seam/snapshot probe command."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    recipe=json.loads((HERE/'recipes/wave-map/recipe.json').read_text())
    command=list(recipe['compile_commands'][0]['command'])
    command.remove(recipe['compile_commands'][0]['original_source']);command.remove('-c')
    k=command.index('-o');del command[k:k+2]
    command.remove('-MD');k=command.index('-MF');del command[k:k+2]
    target=HERE/'probe-build-held';target.mkdir(exist_ok=False)
    source=HERE/'native_seam_and_snapshot.cpp';exe=target/'native-array-probe';dep=target/'native-array-probe.d'
    command.extend([str(source),'-o',str(exe),'-MD','-MF',str(dep)])
    libraries=[]
    for arg in recipe['link_command']:
        if arg.endswith('.a'):
            path=(Path(recipe['link_cwd'])/arg).resolve();libraries.append(path);command.append(str(path))
        elif arg.startswith('-Wl,'):command.append(arg)
    assert len(libraries)==4 and '-include' not in command
    pin_paths=[source,HERE/'include/native_wave_map.hpp',HERE/'recipes/wave-map/include/native_wave_map.hpp',
        HERE/'recipes/wave-map/include/reference_wave_map.hpp',HERE/'recipes/wave-map/include/z4c/hyperboloidal/cartesian_patch.hpp',
        ROOT/'src/z4c/z4c_hyperboloidal.cpp',ROOT/'src/z4c/hyperboloidal/athenak_bridge.hpp',
        ROOT/'build-layer-research/time-projection-controls/rst-reader-gate/restart_reader.py',
        ROOT/'build-layer-research/time-projection-controls/rst-reader-gate/abi.json']+libraries
    r={'scope':'HELD source-only probe. No time step or operator. Separate root source release required.',
       'command':command,'cwd':str(ROOT),'executable':str(exe),'depfile':str(dep),
       'source_before':{str(p):sha(p) for p in pin_paths},
       'compiled_implementation':recipe['compiled_implementation'],
       'prepare_script_sha256':sha(Path(__file__)),
       'source_preparation_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'native_build_receipts':{},'scientific_execution_authorized':False,
       'seam_command':[str(exe),'--seam'],'snapshot_command':[str(exe),'--snapshot','N16|24|32'],
       'snapshot_stdin':'exact little-endian binary64 RST25-field LayoutRight fullarray, no wrapper header',
       'snapshot_stdout':'one finite JSON summary, no evolution'}
    for m in ['wave-map','c0','wave-map-half']:
        p=HERE/'build-attempts'/(m+'-001')/'receipt.json';d=json.loads(p.read_text())
        assert d['passed_compile_link'] and not d['native_executed']
        assert sha(Path(d['executable']))==d['executable_sha256']
        r['native_build_receipts'][m]={'path':str(p),'sha256':sha(p),
            'executable':d['executable'],'executable_sha256':d['executable_sha256']}
    (target/'recipe.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print('PASS held source-only native-array probe recipe')
if __name__=='__main__':main()
