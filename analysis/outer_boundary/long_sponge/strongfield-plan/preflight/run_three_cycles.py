"""Explicitly authorized local three-cycle preflight; never submits jobs."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;ap=argparse.ArgumentParser();ap.add_argument('case',choices=['zero','pulse']);a=ap.parse_args();exe=p.parents[2]/'bin/athena-active-stencil-mpi';expected='67ff395e3c7af43425ef13fe4f0ebecef00256db9450f6ac912f5ab39620f1f8';assert hashlib.sha256(exe.read_bytes()).hexdigest()==expected
run=p/a.case;run.mkdir(exist_ok=False);source=p.parent/'inputs'/('zero_sponge.athinput'if a.case=='zero'else'pulse_sponge.athinput');deck=run/'input.athinput';deck.write_bytes(source.read_bytes());cmd=['mpiexec','-n','8',str(exe),'-i',str(deck),'-t','00:05:00','time/nlim=3','time/ndiag=1'];env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');start=time.monotonic()
with(run/'run.log').open('w')as stream:r=subprocess.run(cmd,cwd=run,env=env,stdout=stream,stderr=subprocess.STDOUT)
rec={'case':a.case,'command':cmd,'binary':str(exe),'binary_sha256':expected,'input_sha256':hashlib.sha256(deck.read_bytes()).hexdigest(),'exit_code':r.returncode,'application_elapsed_seconds':time.monotonic()-start,'scope':'Only three RK3 cycles on eight local MPI ranks; no Aurora submission.'};(run/'execution.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec),flush=True)
raise SystemExit(r.returncode)
