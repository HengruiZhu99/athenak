import pathlib,json,time,subprocess,shutil
r=pathlib.Path('build-performance').resolve();start=time.monotonic()
while not (r/'final-check/completed.json').exists():
 if time.monotonic()-start>3600:raise RuntimeError('final checks incomplete')
 time.sleep(5)
plan=json.loads((r/'cpu-m4-plan.json').read_text());plan['runs']=[x for x in plan['runs'] if x['name'].startswith('setup-fixed-p4-r2-')]
(r/'cpu-m4-replacement-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
subprocess.run(['python3','tst/hyperbolic_shift_3d/performance/run_plan.py',str(r/'cpu-m4-replacement-plan.json'),str(r/'cpu-m4-replacement')],check=True)
old=r/'cpu-m4-bench';new=r/'cpu-m4-replacement';out=r/'cpu-m4-clean';out.mkdir(exist_ok=False)
assert json.loads((old/'binaries.json').read_text())==json.loads((new/'binaries.json').read_text())
original=json.loads((old/'results.json').read_text());replacements={x['case']['name']:x for x in json.loads((new/'results.json').read_text())};selected=[]
for x in original:
 name=x['case']['name'];source=new if name in replacements else old;selected.append(replacements.get(name,x))
 for p in source.glob(name+'.*'):
  if p.suffix not in ['.checkpoint','.bin']:shutil.copy2(p,out/p.name)
(out/'results.json').write_text(json.dumps(selected,indent=2)+'\n');shutil.copy2(old/'binaries.json',out/'binaries.json')
(out/'selection.json').write_text(json.dumps({'original':str(old),'replacement':str(new),'excluded':[x for x in original if x['case']['name'] in replacements],'reason':'Analysis dependency installation overlapped final candidate sample: 2026-11-04T00:16:05.535384Z to 00:16:07.876975Z. Entire pair replaced, both raw runs retained.','selected_names':[x['case']['name'] for x in selected]},indent=2)+'\n')
(out/'completed.json').write_text(json.dumps({'derived_analysis_view':True,'selected_runs':len(selected)})+'\n')
subprocess.run(['python3','tst/hyperbolic_shift_3d/performance/analyze.py',str(out),'--output',str(r/'evidence/cpu-m4-summary.json')],check=True)
subprocess.run(['python3','tst/hyperbolic_shift_3d/performance/check_histories.py',str(out),'--output',str(r/'evidence/cpu-m4-histories.json')],check=True)
