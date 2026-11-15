#!/usr/bin/env python3
"""Profile selected ranks of the frozen solver; never use these runs as throughput."""
import hashlib,json,os,socket,sys
from pathlib import Path
folder=Path(sys.argv[1]);exe=Path(sys.argv[2]);args=sys.argv[3:]
ranks={int(os.environ[k]) for k in ('PMI_RANK','PALS_RANKID') if k in os.environ}
assert len(ranks)==1
rank=ranks.pop();local=int(os.environ.get('MPI_LOCALRANKID',os.environ.get('PALS_LOCAL_RANKID','-1')))
assert 0<=rank<24 and 0<=local<12
assert os.environ['ZE_AFFINITY_MASK']==str(local//2)+'.'+str(local%2)
assert os.environ.get('EnableImplicitScaling')=='0'
d=folder/'ranks'/str(rank);d.mkdir(parents=True)
command=[str(exe),*args]
selected=rank in (0,8,22,23)
if selected:
 tool=Path('/opt/aurora/26.26.0/spack/unified/1.1.1/install/linux-x86_64/pti-gpu-0.17.0-hi6zrvx/bin/unitrace')
 digest=hashlib.sha256(tool.read_bytes()).hexdigest()
 assert digest=='af42167b3842eaaffacd99a7be56aaf6cbcb5f8b6df259f9e3a7bb4eb1f9e17c'
 command=[str(tool),'--host-timing','--device-timing','--kernel-submission','--chrome-call-logging',
  '--chrome-kernel-logging','--chrome-sycl-logging','--chrome-mpi-logging','--verbose','--pid','--tid',
  '--chrome-event-buffer-size','4096','--output-dir-path',str(d),'--output','unitrace.txt',*command]
(d/'launch.json').write_text(json.dumps({'rank':rank,'local_rank':local,'host':socket.gethostname(),'profiled':selected,'command':command,'tool_sha256':digest if selected else None,'scope':'instrumented attribution only, not throughput'},indent=2)+'\n')
os.execv(command[0],command)
