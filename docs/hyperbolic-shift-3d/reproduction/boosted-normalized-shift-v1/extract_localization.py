"""Read-only BN1 endpoint slices using the preserved original-operator BC2 reader.
No campaign source, binary, checkpoint, or evolution settings are modified.
"""
from pathlib import Path
import hashlib,json,shlex,subprocess
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists())
OUT=HERE/'convergence-localization';OUT.mkdir(exist_ok=True)
BUILD=ROOT/'output/boosted-normalized-shift/localization';BUILD.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
target='hyperbolic_shift_3d_checkpoint_constraints_native_control_broad_smooth_six_shells_boosted_free_gauge_refined_core_matched_cfl_normalized_shift'
work=ROOT/'build-boosted-refined-core-v1/tst/hyperbolic_shift_3d'
flags=work/'CMakeFiles'/f'{target}.dir/flags.make'
values={line.split(' = ',1)[0]:line.split(' = ',1)[1] for line in flags.read_text().splitlines() if ' = ' in line}
source=HERE.parent/'boosted-refined-core-v1/endpoint_slices.cpp'
binary=BUILD/'bn1_endpoint_slices'
cmd=['/usr/bin/c++']+shlex.split(values['CXX_DEFINES']+' '+values['CXX_INCLUDES']+' '+values['CXX_FLAGS'])+['-I'+str(ROOT/'tst/hyperbolic_shift_3d'),str(source),'-o',str(binary)]
link=shlex.split((flags.parent/'link.txt').read_text());cmd+=link[link.index('libhyperboloidal_fastflow.a'):]
prov={'compile_command':cmd,'compile_cwd':str(work),'flags':flags.read_text(),'source_sha256':sha(source),'reader_source':str(source.relative_to(ROOT)),'coupled_source_sha256':sha(ROOT/'tst/hyperbolic_shift_3d/coupled_mpi.cpp'),'runs':[]}
if binary.exists():
 old=read(OUT/'slice-provenance.json')
 assert old['binary_sha256']==sha(binary)
 for k in ['flags','source_sha256','coupled_source_sha256']:assert old[k]==prov[k]
if not binary.exists():
 with (OUT/'build.log').open('w') as f:subprocess.run(cmd,cwd=work,stdout=f,stderr=subprocess.STDOUT,check=True)
prov['binary_sha256']=sha(binary)
raw=ROOT/'output/boosted-normalized-shift/aurora'
for n,seg,dt in [(16,1,'.00025'),(20,2,'.0002'),(24,2,str(1/6000))]:
 camp=raw/f'campaigns/n{n}-boosted-normalized-shift-v1-s{n}-evolution-segment{seg:02}'
 a=read(camp/'checkpoint-verified.json');assert a['verified'] and a['cases'][0]['completed']
 cp=raw/a['cases'][0]['checkpoint'];assert sha(cp)==a['cases'][0]['sha256']
 prefix=OUT/f'n{n}';cmd=[str(binary),str(n),str(cp),dt,str(prefix)]
 if not prefix.with_suffix('.json').exists():
  with (OUT/f'n{n}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 assert sha(cp)==a['cases'][0]['sha256']
 meta=read(prefix.with_suffix('.json'));assert meta['schema']=='HYP3D_MPI_BN1' and abs(meta['t_over_M']-10)<1e-10 and meta['native_volume_integral_relative_difference']<1e-9
 prov['runs'].append({'command':cmd,'checkpoint_sha256':sha(cp),'admission_sha256':sha(camp/'checkpoint-verified.json'),'csv_sha256':sha(prefix.with_suffix('.csv')),'verification':meta})
 (OUT/'slice-provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
 print(n,meta,flush=True)
