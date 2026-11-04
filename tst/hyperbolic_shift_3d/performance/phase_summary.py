#!/usr/bin/env python3
"""Use one actual critical-wall rank for each exclusive phase decomposition."""
import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('summary');p.add_argument('--output',required=True);a=p.parse_args();summary=json.loads(pathlib.Path(a.summary).read_text());out=[]
for r in summary['runs']:
 if not r.get('rank_phases'):continue
 ranks=r['rank_phases'];critical=max(range(len(ranks)),key=lambda i:sum(x['seconds'] for x in ranks[i].values()));phases=ranks[critical]
 setup=sum(v['seconds'] for k,v in phases.items() if k in ['startup','native_setup','evolution_setup','initial_state']);regrid=sum(v['seconds'] for k,v in phases.items() if k.startswith('regrid_'));total=sum(v['seconds'] for v in phases.values());steady=r.get('critical_rank_steady_step_with_diagnostics_seconds')
 out.append({'name':r['name'],'critical_wall_rank':critical,'exclusive_phases':phases,'sum_exclusive_seconds':total,'startup_setup_seconds':setup,'all_regrid_seconds':regrid,'setup_and_regrid_fraction':(setup+regrid)/total,'steady_step_with_diagnostics_seconds':steady,'startup_equivalent_steady_steps':setup/steady if steady else None,'caveat':'Phase components all use this same rank. Steady-window critical rank may differ. Short synthetic event history does not imply long-run speedup.'})
pathlib.Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
