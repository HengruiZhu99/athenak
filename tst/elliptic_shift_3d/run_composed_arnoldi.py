#!/usr/bin/env python3
"""One fixed flat 80-step AP Arnoldi diagnostic, independent singular witnesses."""
import argparse,gzip,hashlib,json,os,resource,shlex,signal,subprocess,tarfile,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path.cwd();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);build=root/'build-shift-3d';resource.setrlimit(resource.RLIMIT_CORE,(0,0))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m={'scope':'Unadopted fixed flat N6 seven-shell full-action AP Arnoldi controls; no spectral-gap or scientific-completion claim','source_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'runs':{},'cases':{}}
def save():(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
def run(name,command,limit=180):
 start=time.monotonic();r={'command':list(map(str,command)),'maximum_seconds':limit};m['runs'][name]=r;save()
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
 with (out/(name+'.stdout')).open('w') as stdout,(out/(name+'.stderr')).open('w') as stderr:
  process=subprocess.Popen(r['command'],stdout=stdout,stderr=stderr,env=env,start_new_session=True);r['owned_process_group']=process.pid;save()
  try:r['exit_code']=process.wait(timeout=limit)
  except subprocess.TimeoutExpired:
   r['exit_code']=124
   if os.getpgid(process.pid)!=process.pid:raise RuntimeError('unowned process group')
   os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
 r['seconds']=time.monotonic()-start;save();print(name,r['exit_code'],round(r['seconds'],3),flush=True)
 if r['exit_code']:raise RuntimeError(name+' failed')
flags=(build/'CMakeFiles/elliptic_shift_3d_composite_regrid_tests.dir/flags.make').read_text().splitlines();args=[]
for key in ('CXX_DEFINES =','CXX_INCLUDES ='):args+=shlex.split(next(x.split('=',1)[1] for x in flags if x.startswith(key)))
source=root/'tst/elliptic_shift_3d/probe_composed_arnoldi.cpp';run('dependencies',['/usr/bin/c++','-std=c++17',*args,'-M',source],60)
deps=[Path(s).resolve() for s in shlex.split((out/'dependencies.stdout').read_text().replace('\\\n',' '))[1:]];frozen=out/'source';project={str(f.relative_to(root)):sha(f) for f in deps+[Path(__file__).resolve()] if f.is_file() and f.is_relative_to(root)}
for relative,digest in project.items():
 dest=frozen/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/relative).read_bytes());assert sha(dest)==digest
with tarfile.open(out/'compiled-source.tar.gz','w:gz') as archive:
 for relative in sorted(project):archive.add(frozen/relative,arcname=relative)
m['project_source_hashes']=project;frozenargs=[token.replace(str(root),str(frozen)) for token in args];frozen_source=frozen/source.relative_to(root);run('frozen-dependencies',['/usr/bin/c++','-std=c++17',*frozenargs,'-M',frozen_source],60)
actual=[Path(s).resolve() for s in shlex.split((out/'frozen-dependencies.stdout').read_text().replace('\\\n',' '))[1:]];leaks=[str(p) for p in actual if p.is_relative_to(root) and not p.is_relative_to(frozen)];assert not leaks,leaks;m['live_project_header_leaks']=leaks
libraries=[build/f'kokkos/{name}/src/libkokkos{name}.a' for name in ('containers','algorithms','core','simd')];m['libraries']=[{'path':str(p),'sha256':sha(p)} for p in libraries];binary=out/'probe';run('build',['/usr/bin/c++','-O3','-std=c++17',*frozenargs,frozen_source,'-o',binary,*libraries],60);m['binary_sha256']=sha(binary);(out/'.gitignore').write_text('/probe\n/source/\n/*.bin\n/*.csv\n')
for which in ('expanded','composed'):
 prefix=out/which;run(which+'-arnoldi',[binary,which,'arnoldi',prefix]);data=json.loads((out/(which+'.H.json')).read_text());H=np.array(data['hessenberg']);small=H[:-1];U,s,Vt=np.linalg.svd(H,full_matrices=False);coeff=Vt[-1];np.savetxt(out/(which+'.coefficients.dat'),coeff,fmt='%.17g');run(which+'-witness',[binary,which,'witness',prefix,out/(which+'.coefficients.dat')]);w=json.loads((out/(which+'.witness.json')).read_text());predicted=float(np.linalg.norm(H@coeff));relative=abs(w['independent_full_APv_norm']-predicted)/max(predicted,1e-300);assert relative<1e-8,relative
 eigen=np.linalg.eigvals(small);sym=np.linalg.eigvalsh((small+small.T)/2);comm=small.T@small-small@small.T;_,sv,Vtv=np.linalg.svd(small,full_matrices=False)
 result={'arnoldi_columns':H.shape[1],'singular_values_of_full_action_on_Arnoldi_space':s.tolist(),'restricted_condition':float(s[0]/s[-1]),'smallest_full_action_singular_value':float(s[-1]),'projected_square_smallest_singular_value':float(sv[-1]),'projected_square_relative_nonnormal_commutator':float(np.linalg.norm(comm)/np.linalg.norm(small)**2),'projected_numerical_range_real_min':float(sym[0]),'projected_numerical_range_real_max':float(sym[-1]),'ritz_eigenvalues_real_imag':[[float(z.real),float(z.imag)] for z in eigen],'independent_APv_norm':w['independent_full_APv_norm'],'predicted_APv_norm':predicted,'fresh_action_relative_agreement':relative,'basis_orthogonality_max':data['orthogonality_max'],'witness_Az_over_z':w['Az_over_z'],'regions':w['regions'],'full_spectral_gap_proved':False}
 # Actual minimax residual inside these 80 Krylov directions for the shared start.
 e=np.zeros(H.shape[0]);e[0]=1;curve=[]
 for j in (16,32,48,64,80):
  if j>H.shape[1]:continue
  y=np.linalg.lstsq(H[:j+1,:j],e[:j+1],rcond=None)[0];curve.append({'columns':j,'unrestarted_minimum_relative_residual':float(np.linalg.norm(e[:j+1]-H[:j+1,:j]@y))})
 result['shared_start_unrestarted_residual_curve']=curve;m['cases'][which]=result;save()
 for suffix in ('.Q.bin','.witness.csv'):
  path=out/(which+suffix);compressed=path.with_name(path.name+'.gz');raw=path.read_bytes();compressed.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0));assert gzip.decompress(compressed.read_bytes())==raw;m.setdefault('compressed_files',[]).append({'original':path.name,'bytes':len(raw),'sha256':sha(path),'compressed':compressed.name,'compressed_sha256':sha(compressed)})
m['libraries_unchanged']=all(sha(Path(r['path']))==r['sha256'] for r in m['libraries']);m['project_sources_unchanged']=all(sha(root/name)==digest for name,digest in project.items());m['binary_unchanged']=sha(binary)==m['binary_sha256'];save();assert m['libraries_unchanged'] and m['project_sources_unchanged'] and m['binary_unchanged'];(out/'analysis.json').write_text(json.dumps(m['cases'],indent=2)+'\n');print(json.dumps(m['cases'],indent=2))
