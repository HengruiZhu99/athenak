#!/usr/bin/env python3
"""Stream selected-rank unitrace data; separate overlapping host/GPU costs."""
import argparse,collections,hashlib,json,re,statistics
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def events(path,kind):
 with path.open() as f:
  for line in f:
   if kind=='gpu' and '"cat": "gpu_op"' not in line:continue
   if kind=='cpu' and '"cat": "cpu_op"' not in line:continue
   if not line.startswith('{"ph"'):continue
   e=json.loads(line.rstrip().rstrip(','))
   if e.get('ph')=='X':yield e

def group(name):
 m=re.search(r'((?:z4c_hyperboloidal|bvals)::[\w:]+)\(',name)
 if m:return m[1]
 if name.startswith('zeCommandList'):return name.split('[')[0]
 return name[:160]

def stats(values):
 return {'calls':len(values),'sum_seconds':sum(values)/1e6,'mean_us':statistics.mean(values),'median_us':statistics.median(values),'max_us':max(values)}

def analyze(path):
 gpu=list(events(path,'gpu'))
 markers=sorted(e['ts'] for e in gpu if 'Z4cHyperboloidal::CopyU(Driver*, int)::{lambda(int)#1}' in e['name'])
 assert len(markers)==8,(path,len(markers))
 begin=markers[1];end=max(e['ts']+e['dur'] for e in gpu if 'Z4cHyperboloidal::ExpRKUpdate(' in e['name'])
 groups=collections.defaultdict(list);full=collections.defaultdict(list)
 for e in gpu:
  if e['ts']>=begin and e['ts']+e['dur']<=end+1:
   groups[group(e['name'])].append(e['dur']);full[e['name']].append(e['dur'])
 cpu=collections.defaultdict(lambda:[0,0.]);io=collections.defaultdict(lambda:[0,0.]);bytes_sent=0;missing_bytes=0
 for e in events(path,'cpu'):
  name=e['name'];dur=e['dur']
  if name.startswith('MPI::MPI_File'):
   io[name][0]+=1;io[name][1]+=dur/1e6
  if e['ts']<begin or e['ts']+dur>end+1:continue
  cpu[name][0]+=1;cpu[name][1]+=dur/1e6
  if name=='MPI::MPI_Isend':
   size=e.get('args',{}).get('dsize')
   if size is None:missing_bytes+=1
   else:bytes_sent+=size
 return {'trace':str(path),'trace_sha256':sha(path),'window_start_us':begin,'window_end_us':end,
  'window_span_seconds':(end-begin)/1e6,'cycles':7,'MPI_Isend_reported_dsize':None if missing_bytes else bytes_sent,'MPI_Isend_missing_size_records':missing_bytes,
  'MPI_size_note':'Raw unitrace dsize field, not validated bytes. In v2 agreement-minus-masked traces this is half the native FP64 sparse-plan byte count; native counts determine communication volume.',
  'gpu_groups':{k:stats(v) for k,v in sorted(groups.items(),key=lambda kv:-sum(kv[1]))},
  'gpu_kernel_shapes':{k:stats(v) for k,v in sorted(full.items(),key=lambda kv:-sum(kv[1]))},
  'cpu_window':{k:{'calls':v[0],'sum_seconds':v[1]} for k,v in sorted(cpu.items(),key=lambda kv:-kv[1][1])},
  'whole_run_MPI_IO':{k:{'calls':v[0],'sum_seconds':v[1]} for k,v in io.items()}}

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--modes',nargs='+',default=['legacy','masked','agreement']);a=p.parse_args()
 verified=json.loads((a.run/'results.json').read_text());ready={x['mode'] for x in verified if x['accepted_payload_bitwise_equal']}
 assert all(x in ready for x in a.modes)
 out={'scope':'Instrumented attribution, not throughput; window spans cycles2-8 GPU work, ending at final update; final tracker work is excluded. Host waits and GPU durations overlap and must not be added.','script_sha256':sha(Path(__file__)),'modes':{}}
 for mode in a.modes:
  out['modes'][mode]={}
  for rank in [0,8,22,23]:
   d=a.run/mode/'ranks'/str(rank);paths=[x for x in d.glob('*.json') if x.name!='launch.json'];assert len(paths)==1
   result=analyze(paths[0]);out['modes'][mode][str(rank)]=result
   top=[(k,round(v['sum_seconds']/7,6)) for k,v in result['gpu_groups'].items()][:6]
   print(mode,rank,'GPU seconds/cycle',top,flush=True)
 (a.run/('profile-summary-'+'-'.join(a.modes)+'.json')).write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
