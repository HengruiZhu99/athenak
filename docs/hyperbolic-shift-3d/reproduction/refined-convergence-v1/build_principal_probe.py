"""Separate continuum-principal analysis executable; no campaign mutation."""
import pathlib,shlex,json,subprocess,hashlib,argparse
p=argparse.ArgumentParser();p.add_argument("--amplitude",type=float,default=1);p.add_argument("--mode",choices=["sweep","crossings","separated"],default="crossings");a=p.parse_args()
if a.amplitude not in (1,.25):raise SystemExit("supported reproducibility controls are 1 and .25")
HERE=pathlib.Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists())
OUT=ROOT/'output/refined-convergence/principal-probe';OUT.mkdir(exist_ok=True)
target=ROOT/'build-refined-convergence/tst/hyperbolic_shift_3d';src=HERE/'principal_probe.cpp';flags={}
for line in (target/'CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/flags.make').read_text().splitlines():
 if ' = ' in line:
  k,v=line.split(' = ',1);flags[k]=shlex.split(v)
obj=OUT/'probe.o';binary=OUT/'probe'
cmd=['/usr/bin/c++']+flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+['-I'+str(ROOT/'tst/hyperbolic_shift_3d'),'-c',str(src),'-o',str(obj)]
subprocess.run(cmd,check=True)
link=shlex.split((target/'CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/link.txt').read_text());link[link.index('CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/amr_evolution.cpp.o')]=str(obj);link[link.index('-o')+1]=str(binary);subprocess.run(link,cwd=target,check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/('build-reproduced-'+a.mode+'-'+str(a.amplitude)+'.json')).write_text(json.dumps(dict(compile=cmd,link=link,source_sha256=sha(src),binary_sha256=sha(binary)),indent=2)+'\n')
for name in ['central6_stationary_matched_plateau','central20_stationary_smr025_radial9']:
 suffix='' if a.mode=='sweep' else '-separated' if a.mode=='separated' else ('-crossings' if a.amplitude==1 else '-crossings-quarter')
 cmd=[str(binary),str(ROOT/f'inputs/hyperbolic_shift_3d/refined_convergence/{name}.cfg'),str(OUT/(name+suffix+'.jsonl'))]
 if a.mode=='crossings':cmd+=['crossings',str(a.amplitude)]
 elif a.mode=='separated':cmd+=['separated']
 elif a.amplitude!=1:raise SystemExit('sweep uses amplitude 1')
 subprocess.run(cmd,check=True)
