#!/usr/bin/env python3
"""Summarize matched unprofiled windows, retaining startup and output intervals."""
import argparse,json,re,statistics,hashlib
from pathlib import Path

def breakdown(path):
 rows=[];between=[]
 for line in path.read_text().splitlines():
  m=re.search(r'elapsed=([\deE.+-]+) cycle=(\d+) time=([\deE.+-]+)',line)
  if m:
   rows.append((float(m[1]),int(m[2]),float(m[3]),any('binary horizon' in x for x in between)))
   between=[]
  else:between.append(line)
 last=rows[-1][1];ordinary=[];history=[];final=[]
 for a,b in zip(rows,rows[1:]):
  if b[1]!=a[1]+1:continue
  value=b[0]-a[0]
  if b[1]==last:final.append(value)
  elif b[3]:history.append(value)
  elif a[1]>=20:ordinary.append(value)
 return {'ordinary_post_warmup_calls':len(ordinary),
  'ordinary_median_seconds':statistics.median(ordinary) if ordinary else None,
  'ordinary_mean_seconds':statistics.mean(ordinary) if ordinary else None,
  'ordinary_stdev_seconds':statistics.stdev(ordinary) if len(ordinary)>1 else None,
  'history_bearing_intervals_seconds':history,'final_step_plus_output_seconds':final,
  'caution':'Output-bearing intervals also contain one PDE step; their excess over ordinary time is an estimate, not an instrumented I/O timer.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
 rows=json.loads((a.run/'measurements.json').read_text());by={};details=[]
 for row in rows:
  if row['steps']<100:continue
  d=a.run/('%02d-%s-%d'%(row['index'],row['mode'],row['steps']))
  details.append({**row,'interval_breakdown':breakdown(d/'run.log')})
  by.setdefault(row['mode'],[]).append(row)
 launch=json.loads((a.run/'launch-provenance.json').read_text())
 result={'scope':'Unprofiled matched fresh-initialization windows; partial pilot unless protocol completes. Startup and final outputs are included in end-to-end wall/M0.','protocol':launch['protocol'],'complete_protocol':len(rows)==launch.get('expected_windows',10),'windows':details,'modes':{},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 for mode,items in by.items():
  wall=[x['end_to_end_seconds_per_M0'] for x in items]
  result['modes'][mode]={'windows':len(items),'mean_seconds_per_M0':statistics.mean(wall),'stdev_seconds_per_M0':statistics.stdev(wall) if len(wall)>1 else None,'min_seconds_per_M0':min(wall),'max_seconds_per_M0':max(wall),'median_step_mean_seconds':statistics.mean(x['typical_step_seconds'] for x in items)}
 if 'legacy' in by and 'agreement' in by:
  result['agreement_over_legacy_wall_M0']=result['modes']['agreement']['mean_seconds_per_M0']/result['modes']['legacy']['mean_seconds_per_M0']
  result['within_1p15_mean']=result['agreement_over_legacy_wall_M0']<=1.15
  baseline=result['modes']['legacy'];candidate=result['modes']['agreement']
  result['all_window_ratio_range']=[candidate['min_seconds_per_M0']/baseline['max_seconds_per_M0'],candidate['max_seconds_per_M0']/baseline['min_seconds_per_M0']]
  result['variability_note']='Observed all-window ratio envelope, not a confidence interval. Three windows cannot establish a precise distribution tail.'
  result['performance_within_observed_envelope']=result['complete_protocol'] and result['all_window_ratio_range'][1]<=1.15
  index={r['index']:r for r in rows}
  pairs=[(0,1),(3,2)] if launch.get('expected_windows',10)==4 else [(3,4),(7,6),(9,8)]
  paired=[]
  for bi,ci in pairs:
   if bi not in index or ci not in index:continue
   b,c=index[bi],index[ci]
   assert b['mode']=='legacy' and c['mode']=='agreement' and b['dt']==c['dt'] and b['steps']==c['steps']
   paired.append({'baseline_window':bi,'agreement_window':ci,'wall_M0_ratio':c['end_to_end_seconds_per_M0']/b['end_to_end_seconds_per_M0']})
  result['paired_windows']=paired
  if paired:
   ratios=[r['wall_M0_ratio'] for r in paired]
   result['paired_ratio_range']=[min(ratios),max(ratios)]
   result['paired_ratio_mean']=statistics.mean(ratios)
   result['paired_ratio_sample_stdev']=statistics.stdev(ratios) if len(ratios)>1 else None
   result['all_completed_pairs_below_1p15']=result['complete_protocol'] and max(ratios)<=1.15
  result['performance_assessment']=('passes this protocol within the observed window envelope' if result['performance_within_observed_envelope'] else 'not established for this protocol')
  result['acceptance']='Overall goal additionally requires late-time scientific benefit and matched-resolution checks; this timing report cannot establish those.'
 (a.run/'performance-summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result['modes'],indent=2));print('ratio',result.get('agreement_over_legacy_wall_M0'))
if __name__=='__main__':main()
