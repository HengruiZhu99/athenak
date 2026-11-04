#!/usr/bin/env python3
"""Generate static plots from archived summaries; show every repeat and spread."""
import argparse,json,pathlib,statistics,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('evidence');p.add_argument('output');a=p.parse_args();root=pathlib.Path(a.evidence);out=pathlib.Path(a.output);out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
cpu=json.loads((root/'cpu-m4-summary.json').read_text());gpu=json.loads((root/'gpu2-summary.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,summary,title in zip(axes,[cpu,gpu],['Apple M5 Pro · Serial MPI','Aurora · GPU-aware MPI']):
 groups=[c for c in summary['paired_comparisons'] if c['experiment'].startswith('setup-')]
 for j,c in enumerate(groups):
  m=c['metrics']['process_seconds'];vals=m['speedup_ratios'];ax.scatter([j-.08,j,j+.08],vals,s=45,color='#176b87',zorder=3)
  ax.plot([j-.22,j+.22],[m['median_speedup']]*2,color='#102a43',lw=2)
  lo,hi=m['paired_log_ratio_student_t_95_interval'];ax.errorbar(j,m['median_speedup'],yerr=[[m['median_speedup']-lo],[hi-m['median_speedup']]],color='#176b87',capsize=4,fmt='none')
 ax.set_xticks(range(len(groups)),[c['experiment'].removeprefix('setup-')+' · P='+str(c['ranks']) for c in groups]);ax.axhline(1,color='gray',ls='--');ax.set_title(title);ax.set_ylabel('Whole-process speedup (M0 / direct setup)');ax.set_ylim(bottom=.95);ax.grid(axis='y',alpha=.2)
fig.suptitle('Construct the final halo plan once\nThree alternating pairs; dots are pairs, bars are paired log-ratio 95% t intervals')
fig.savefig(out/'setup-speedup.png');plt.close(fig)
rows=[r for r in gpu['runs'] if r['name'].startswith('scaling-')];groups={}
for r in rows:
 c=r['case'];key=(c['policy'],c['ranks'],c.get('ppn',min(c['ranks'],12)));groups.setdefault(key,[]).append(r)
scaling=[]
for key,rs in sorted(groups.items()):
 policy,P,ppn=key;baseline=groups[(policy,1,1)];ref=statistics.median(r['critical_rank_steady_step_with_diagnostics_seconds'] for r in baseline)
 vals=[r['critical_rank_steady_step_with_diagnostics_seconds'] for r in rs];median=statistics.median(vals)
 scaling.append({'policy':policy,'ranks':P,'ppn':ppn,'used_nodes':math.ceil(P/ppn),'repeats':len(rs),'step_seconds':vals,'median_step_seconds':median,'step_range':[min(vals),max(vals)],'speedup':ref/median,'efficiency':ref/median/P,'process_seconds':[r['process_seconds'] for r in rs],'median_process_seconds':statistics.median(r['process_seconds'] for r in rs),'max_rank_rss_bytes':max(r['max_rank_rss_bytes'] for r in rs),'sum_rank_peak_rss_bytes_upper_bound':max(r['sum_rank_peak_rss_bytes_upper_bound'] for r in rs),'max_node_sum_rank_peak_rss_bytes_upper_bound':max(r['max_node_sum_rank_peak_rss_bytes_upper_bound'] for r in rs)})
(out/'scaling.json').write_text(json.dumps(scaling,indent=2)+'\n')
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for policy,color,label in [('fixed','#176b87','Mixed: 15 core + 36 shells'),('uniform','#ba4a00','Uniform: 64 core + 36 shells')]:
 points=[s for s in scaling if s['policy']==policy and not (s['ranks']==8 and s['ppn']==4)];x=[s['ranks'] for s in points]
 axes[0].errorbar(x,[s['median_step_seconds'] for s in points],yerr=[[s['median_step_seconds']-s['step_range'][0] for s in points],[s['step_range'][1]-s['median_step_seconds'] for s in points]],marker='o',color=color,label=label,capsize=3)
 axes[1].plot(x,[s['speedup'] for s in points],'-o',color=color,label=label)
 axes[2].plot(x,[s['max_rank_rss_bytes']/2**30 for s in points],'-o',color=color,label=label)
for s in scaling:
 if s['ranks']==8 and s['ppn']==4:
  for ax,y in zip(axes,[s['median_step_seconds'],s['speedup'],s['max_rank_rss_bytes']/2**30]):ax.scatter([8],[y],marker='D',color='#674ea7',s=45,zorder=5,label='Mixed: 8 ranks across 2 nodes')
for ax in axes:ax.set_xlabel('MPI ranks / GPU tiles');ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,12,24],[1,2,4,8,12,24]);ax.grid(alpha=.2)
axes[0].set_ylabel('Critical-rank seconds / step + diagnostics');axes[1].set_ylabel('Steady speedup relative to same-problem P=1');axes[2].set_ylabel('Maximum rank host peak RSS (GiB)');axes[0].legend(fontsize=8)
fig.suptitle('Aurora strong scaling · direct setup · 12 tiles/node · N12, dt=0.00025, 32 steps\nWarmup: 4 steps; three repeats; timing error bars show full range; RSS excludes device memory')
fig.savefig(out/'strong-scaling.png');plt.close(fig)
