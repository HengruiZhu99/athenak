#!/usr/bin/env python3
"""User-authorized low-resolution local Kerr scalar exploration; no scheduler."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_cpu import run
from scalar_evidence import waveform, history


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


OBSERVER = r'''
    // BEGIN PASSIVE LOCAL SCRI OBSERVER: reads only the host state mirror.
    if(p.cap==1) {
      double zr=0,zi=0,vr=0,vi=0,gr=0,gi=0;
      const double norm=std::sqrt(15/(32*std::acos(-1.)));
      for(int b=1;b<7;++b) {auto z=blocks[b];
        for(int k=NG;k<NG+z.n[2];++k)for(int j=NG;j<NG+z.n[1];++j) {
          int l=z.index(NG+z.n[0]-1,j,k);auto c=host[l];double n[3];
          for(int a=0;a<3;++a)n[a]=c.physical.x[a]/p.S;
          double area=z.dx[1]*z.dx[2]*std::pow(std::max({fabs(n[0]),fabs(n[1]),fabs(n[2])}),3);
          double re=norm*(n[0]*n[0]-n[1]*n[1]),im=-2*norm*n[0]*n[1];
          double v=h(l,1);if(p.normal_upwind) {v=-h(l,1)/std::sqrt(-c.physical.g[0][0]);for(int a=0;a<3;++a)v+=-c.physical.g[0][a+1]/c.physical.g[0][0]*gradient(h,l,0,a,c);}
          zr+=area*h(l,0)*re;zi+=area*h(l,0)*im;vr+=area*v*re;vi+=area*v*im;
          gr+=area*re*re;gi+=area*im*im;
        }
      }
      bool header=!std::filesystem::exists(output+"/waveform-scri.csv");
      std::ofstream scri(output+"/waveform-scri.csv",std::ios::app);scri<<std::setprecision(17);
      if(header)scri<<"time,real22,imag22,raw_real22,raw_imag22,vreal22,vimag22,gram_real,gram_imag\n";
      scri<<t<<","<<p.S*zr/(2*gr)<<","<<p.S*zi/(2*gi)<<","<<p.S*zr<<","<<p.S*zi<<","<<p.S*vr/(2*gr)<<","<<p.S*vi/(2*gi)<<","<<gr<<","<<gi<<"\n";
    }
    // END PASSIVE LOCAL SCRI OBSERVER
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();E=a.evidence.resolve();R=a.output.resolve();repo=Path(__file__).resolve().parents[2];scripts=Path(__file__).resolve().parent
    if R.exists():raise RuntimeError('Existing local exploration must be inspected; no duplicates')
    R.mkdir();source=R/'source';source.mkdir()
    manifest=read(E/'cpu-v22/source-manifest.json')
    if any(sha(repo/name)!=digest for name,digest in manifest.items()):raise RuntimeError('Preserved scalar source changed')
    for name in ['main.cpp','geometry.hpp','sbp63.hpp']:
        shutil.copy2(repo/'src/test_scalar'/name,source/name)
    original=(source/'main.cpp').read_text()
    marker='    // Fixed rho=10, hence fixed physical r=10/Omega(10) within each map family.'
    if original.count(marker)!=1:raise RuntimeError('Observer insertion is ambiguous')
    instrumented=original.replace(marker,OBSERVER+'\n'+marker)
    if instrumented.replace(OBSERVER+'\n','',1)!=original:raise RuntimeError('Observer changed evolution code')
    (source/'main.cpp').write_text(instrumented)
    common=['--mass','1','--spin','.5','--cap','1','--excision','0','--boundary','radiation','--data','ringdown','--normal-upwind','1','--outer-sbp','2','--ko','0','--reduction-damping','0','--smooth-layer','1','--radial-stretch','-2','--cfl','.18']
    plan=dict(user_authorization=['proceed to the spinning case, perhaps with a spin of0.5','just run low resolution exploration locally for now'],
              exploration_only=True,previous_Schwarzschild_admission=False,resolutions=[12,16,20],duration=100,spin=.5,threads=4,
              common_arguments=common,wall_checkpoint_seconds=300,maximum_segments_per_resolution=8,
              source_sha256=manifest,observer_source_sha256=sha(source/'main.cpp'),observer_removal_reproduces_original=True,
              original_source_sha256=sha(repo/'src/test_scalar/main.cpp'),
              checks='Finite evolution, unchanged four frozen caches, raw checkpoints, free two-branch QNM fits, comparison to exploratory independent600x8 solution. No formal three-grid or GPU admission.',
              scri_diagnostic='Every0.5M: raw angular projection and diagonal real/imag harmonic Gram normalization; radiation rPhi=Spsi. Normalization reduces l2 quadrature bias and does not remove physical higher-l mixing.',
              remote_policy=dict(now='local_only',future_queue='capacity',future_nodes=1,future_walltime='00:30:00',future_project='CompactBinaryMerger',maximum_allocations_queued_or_running=1),
              declared_at=time.time())
    (R/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    exe=R/'test_scalar'
    subprocess.run([sys.executable,str(scripts/'build_audit_executable.py'),'--build',str(repo/'build-test-scalar-omp'),'--source',str(source/'main.cpp'),'--output',str(exe)],check=True)
    os.environ['SCALAR_CPU_THREADS']='4';os.environ['SCALAR_CPU_PROC_BIND']='false';os.environ['OMP_DYNAMIC']='FALSE'
    # Initialization-only comparison: no duplicate trajectory is evolved.
    original_run=run(E/'cpu-v22/test_scalar',R,'observer-original-initialization',['--n','12','--tlim','0',*common])
    observed_run=run(exe,R,'observer-instrumented-initialization',['--n','12','--tlim','0',*common])
    observer_check=dict(no_time_steps=True,checkpoint_bitwise_equal=sha(original_run/'checkpoint.f64')==sha(observed_run/'checkpoint.f64'),frozen_cache_equal=read(original_run/'frozen-integrity.json')==read(observed_run/'frozen-integrity.json'))
    (R/'observer-review.json').write_text(json.dumps(observer_check,indent=2)+'\n')
    if not observer_check['checkpoint_bitwise_equal'] or not observer_check['frozen_cache_equal']:raise RuntimeError('Passive observation altered state')
    # Independent oracle is a separate small single-core local process.
    oracle=E/'kerr-oracle';oracle_exe=oracle/'kerr_oracle-checkpoint-v2-clang';table=oracle/'coefficients-spin05-n600-a8.txt'
    preflight=read(oracle/'stable-oracle-source-preflight.json');meta=read(table.with_suffix('.json'))
    if sha(oracle_exe)!=preflight['binary_sha256'] or sha(table)!=meta['table_sha256']:raise RuntimeError('Independent oracle preflight mismatch')
    reference=R/'independent';reference.mkdir();refwave=reference/'waveform.csv';refcmd=[str(oracle_exe),str(table),str(refwave),'100']
    (reference/'invocation.json').write_text(json.dumps(dict(command=refcmd,binary_sha256=sha(oracle_exe),table_sha256=sha(table),scope='Exploratory600x8 independent solution only; no reference-resolution admission.'),indent=2)+'\n')
    refstart=time.time();reflog=(reference/'run.log').open('w');refproc=subprocess.Popen(refcmd,stdout=reflog,stderr=subprocess.STDOUT)
    print('LOCAL INDEPENDENT KERR PID',refproc.pid,flush=True)
    finished={}
    for n in plan['resolutions']:
        current=run(exe,R,f'kerr-n{n}',['--n',str(n),'--tlim','100',*common,'--wall-seconds','300'])
        for segment in range(1,8):
            if (current/'complete.txt').exists():break
            if not (current/'checkpointed.txt').exists():raise RuntimeError('No confirmed local checkpoint')
            child=R/f'kerr-n{n}-resume{segment}'
            subprocess.run([sys.executable,str(scripts/'resume_cpu.py'),'--exe',str(exe),'--parent',str(current),'--output',str(child),'--tlim','100','--wall-seconds','300'],check=True)
            current=child
        if not (current/'complete.txt').exists():raise RuntimeError('Bounded local exploration budget reached; checkpoint preserved')
        finished[n]=str(current)
        (R/'progress.json').write_text(json.dumps(dict(finished=finished,updated=time.time(),next_resolution=next((q for q in plan['resolutions'] if q not in finished),None)),indent=2)+'\n')
        subprocess.run([sys.executable,str(scripts/'fit_ringdown.py'),str(current/'waveform.csv'),'--spin','.5','--expected',str(E/'spin-zero-qnm-both-branches.json'),'--output',str(R/f'qnm-n{n}.json')],check=True)
        print('LOCAL KERR COMPLETE',n,flush=True)
    rc=refproc.wait();reflog.close();(reference/'receipt.json').write_text(json.dumps(dict(returncode=rc,elapsed_seconds=time.time()-refstart,files={f.name:sha(f) for f in reference.iterdir() if f.is_file()}),indent=2)+'\n')
    if rc:raise RuntimeError('Independent exploratory oracle failed; evidence preserved')
    subprocess.run([sys.executable,str(scripts/'fit_ringdown.py'),str(refwave),'--spin','.5','--expected',str(E/'spin-zero-qnm-both-branches.json'),'--output',str(R/'qnm-independent.json')],check=True)
    (R/'completion.json').write_text(json.dumps(dict(exploration_complete=True,formal_validation=False,finished=finished,independent=str(refwave),plan_sha256=sha(R/'plan.json'),observer_review_sha256=sha(R/'observer-review.json'),binary_sha256=sha(exe),completed_at=time.time()),indent=2)+'\n')


if __name__=='__main__':main()
