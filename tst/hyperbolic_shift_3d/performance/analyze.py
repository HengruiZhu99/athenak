#!/usr/bin/env python3
"""Summarize completed receipts, critical-rank phases and paired variability."""
import argparse,json,pathlib,re,statistics,math
p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',required=True);a=p.parse_args();root=pathlib.Path(a.directory)
receipts=json.loads((root/'results.json').read_text());rows=[]
for receipt in receipts:
    case=receipt['case'];name=case['name'];row={'name':name,'case':case,'passed':receipt['passed'],'process_seconds':receipt['seconds']}
    if receipt['returncode']!=0:rows.append(row);continue
    report=json.loads((root/(name+'.report.json')).read_text());row['rank0_after_setup_seconds']=report['seconds']
    profiles=[json.loads(f.read_text()) for f in sorted(root.glob(name+'.rank*.performance.json'))]
    if profiles:
        assert len(profiles)==case['ranks']
        row['max_rank_driver_seconds']=max(x['wall_seconds'] for x in profiles)
        row['max_rank_rss_bytes']=max(x['peak_rss_bytes'] for x in profiles)
        row['sum_rank_peak_rss_bytes_upper_bound']=sum(x['peak_rss_bytes'] for x in profiles)
        row['rank_phases']=[x['exclusive_phases'] for x in profiles]
        row['critical_rank_step_seconds']=max(x['exclusive_phases'].get('step',{}).get('seconds',0) for x in profiles)
        windows=[];step_windows=[]
        # Fixed hierarchy only: diagnostics are initial, then one per step.
        if case.get('policy','fixed') in ['fixed','uniform']:
            for f in sorted(root.glob(name+'.rank*.samples.jsonl')):
                samples=[json.loads(line) for line in f.read_text().splitlines()]
                steps=[x['seconds'] for x in samples if x['phase']=='step']
                diag=[x['seconds'] for x in samples if x['phase']=='diagnostics']
                warm=4 if len(steps)>=16 else 2
                assert len(steps)==case['steps'] and len(diag)==len(steps)+1
                step_windows.append(statistics.mean(steps[warm:]))
                windows.append(statistics.mean([s+d for s,d in zip(steps[warm:],diag[warm+1:])]))
            if windows:
                row['critical_rank_steady_step_seconds']=max(step_windows)
                row['critical_rank_steady_step_with_diagnostics_seconds']=max(windows)
                row['warmup_steps']=warm
        generations=[]
        for f in sorted(root.glob(name+'.rank*.generations.jsonl')):
            generations.append([json.loads(line) for line in f.read_text().splitlines() if json.loads(line)['event'] in ['retired','final']])
        row['rank_generation_totals']=[{'nested_seconds':{k:sum(g['nested_seconds'].get(k,0) for g in gs) for k in gs[-1]['nested_seconds']},'sparse_bytes_sent':sum(g['sparse_calls']*g['sparse_send_bytes_per_call'] for g in gs)} for gs in generations]
    rows.append(row)
groups={}
for row in rows:
    match=re.fullmatch(r'(.*)-p(\d+)-r(\d+)-(base|m0|m1a|m1b|m4)',row['name'])
    if match:
        label,ranks,repeat,variant=match.groups();groups.setdefault((label,int(ranks)),{}).setdefault(int(repeat),{})[variant]=row
comparisons=[]
for (label,ranks),pairs in groups.items():
    variants=set(v for pair in pairs.values() for v in pair);control='base' if 'base' in variants else 'm0'
    candidate=next(v for v in variants if v!=control)
    result={'experiment':label,'ranks':ranks,'control':control,'candidate':candidate,'metrics':{}}
    for metric in ['process_seconds','rank0_after_setup_seconds','critical_rank_step_seconds','critical_rank_steady_step_seconds','critical_rank_steady_step_with_diagnostics_seconds']:
        ratios=[pair[control][metric]/pair[candidate][metric] for pair in pairs.values() if control in pair and candidate in pair and metric in pair[control] and metric in pair[candidate]]
        if not ratios:continue
        m={'pairs':len(ratios),'speedup_ratios':ratios,'median_speedup':statistics.median(ratios),'range':[min(ratios),max(ratios)]}
        if len(ratios)==3:
            logs=list(map(math.log,ratios));r=.95;t=math.sqrt(2*r*r/(1-r*r));margin=t*statistics.stdev(logs)/math.sqrt(3)
            m['paired_log_ratio_student_t_95_interval']=[math.exp(statistics.mean(logs)-margin),math.exp(statistics.mean(logs)+margin)]
        result['metrics'][metric]=m
    comparisons.append(result)
pathlib.Path(a.output).write_text(json.dumps({'directory':str(root),'completed':(root/'completed.json').exists(),'runs':rows,'paired_comparisons':comparisons,'notes':['Profiled runs excluded from A/B groups.','Rank-zero report excludes initial setup but includes final capture/write.','Rank RSS sum is an upper bound, not simultaneous measured node RSS.','Three-pair Student-t intervals assume independent log ratios; workstation scheduling limits inference.']},indent=2)+'\n')
