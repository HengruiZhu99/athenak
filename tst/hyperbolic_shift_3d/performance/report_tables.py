#!/usr/bin/env python3
import argparse,json,pathlib,statistics
p=argparse.ArgumentParser();p.add_argument('evidence');p.add_argument('plots');p.add_argument('--output',required=True);a=p.parse_args();root=pathlib.Path(a.evidence);lines=['## Repeated setup comparisons','', '| Backend / case | Ranks | M0 process (s) | M4 process (s) | Paired median speedup | Paired 95% t interval |','|---|---:|---:|---:|---:|---:|']
for backend,file in [('CPU','cpu-m4-summary.json'),('Aurora','gpu2-summary.json')]:
 s=json.loads((root/file).read_text())
 for c in s['paired_comparisons']:
  if not c['experiment'].startswith('setup-'):continue
  m=c['metrics']['process_seconds'];lo,hi=m['paired_log_ratio_student_t_95_interval'];rs=[r for r in s['runs'] if r['name'].startswith(c['experiment']+'-p')];b=statistics.median(r['process_seconds'] for r in rs if r['case']['variant']=='m0');d=statistics.median(r['process_seconds'] for r in rs if r['case']['variant']=='m4')
  lines.append(f"| {backend} / {c['experiment'][6:]} | {c['ranks']} | {b:.3f} | {d:.3f} | {m['median_speedup']:.3f}× | {lo:.3f}–{hi:.3f} |")
lines+=['','Three alternating pairs per row. Fixed: eight steps, 15 Cartesian leaves;','dynamic: six steps and 15→22→8 leaves. The same six-shell exterior and','every-step diagnostics are retained. The intervals assume independent log','ratios and do not remove workstation scheduling uncertainty.','', '![Setup comparisons](plots/setup-speedup.png)','', '## Aurora strong scaling','', '| Problem | Ranks / used nodes | Step + diagnostics (s) | Full range (s) | Steady speedup / efficiency | Process (s) | Max rank host RSS (GiB) | Max node RSS upper bound (GiB) |','|---|---:|---:|---:|---:|---:|---:|---:|']
rows=json.loads((pathlib.Path(a.plots)/'scaling.json').read_text())
for s in rows:
 lo,hi=s['step_range'];label='Mixed' if s['policy']=='fixed' else 'Uniform';lines.append(f"| {label} | {s['ranks']} / {s['used_nodes']} | {s['median_step_seconds']:.5f} | {lo:.5f}–{hi:.5f} | {s['speedup']:.2f}× / {100*s['efficiency']:.1f}% | {s['median_process_seconds']:.2f} | {s['max_rank_rss_bytes']/2**30:.2f} | {s['max_node_sum_rank_peak_rss_bytes_upper_bound']/2**30:.2f} |")
lines+=['','Each row has three repeats of the same 32-step problem; four warmup steps','are excluded from the critical-rank steady window. Speedup uses that problem’s','one-rank median. Both nodes were allocated for this sweep; “used nodes” follows','the explicit ranks-per-node launch. The two eight-rank mixed rows isolate','placement. Node memory is the sum of individual rank peaks under the verified','PALS contiguous placement, an upper bound rather than simultaneous RSS.','', '![Strong scaling](plots/strong-scaling.png)']
pathlib.Path(a.output).write_text('\n'.join(lines)+'\n')
