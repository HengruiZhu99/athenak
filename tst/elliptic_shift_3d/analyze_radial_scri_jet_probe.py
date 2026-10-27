#!/usr/bin/env python3
"""All-row radial scri diagnostics and explicit failed scientific gates."""
import argparse,csv,gzip,hashlib,json,math,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=Path.cwd();src=a.input.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);os.environ['MPLCONFIGDIR']=str(out/'mpl-cache')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();manifest=json.loads((src/'manifest.json').read_text());data={};failures={}
for name,run in manifest['runs'].items():
 if name.endswith(('-build','-deps')):continue
 if run['exit_code']:failures[name]=dict(exit_code=run['exit_code'],reason=(src/(name+'.stderr')).read_text().strip());continue
 f=src/(name+'.stdout')
 if f.stat().st_size:data[name]=json.loads(f.read_text())
orders=[]
for layers,ns in [(1,[6,12,18]),(2,[6,10,14]),(7,[6,8,10])]:
 for points in [6,7]:
  rows=[data[f'mms-{layers}-1-{n}-{points}-log'] for n in ns]
  item=dict(layers=layers,nodes=points,n=ns)
  for key in ['radial_AD_first_max','radial_AD_second_max']:
   vals=[row[key] for row in rows];item[key]=vals;item[key+'_orders']=[math.log(vals[i]/vals[i+1])/math.log(ns[i+1]/ns[i]) for i in range(2)]
  for region in ['all_surface','edge_band','corner_band','exact_patch_seams_and_corners']:
   item[region]={key:[row[region][key] for row in rows] for key in ['value_max','first_max','second_max','normal_tangent_hessian_max']}
  orders.append(item)
pointwise={};compressed=[]
for f in sorted(src.glob('pole-*.csv')):
 if not f.stat().st_size:continue
 rows=list(csv.DictReader(f.open()));evolved=[x for x in rows if x['state']=='1'];initial=[x for x in rows if x['state']=='0'];allpole=['chi','Khat','Theta','alpha','Axx','Axy','Axz','Ayy','Ayz','Azz','LambdaX','LambdaY','LambdaZ']
 def maximum(rows):
  point,field=max(((row,key) for row in rows for key in allpole),key=lambda x:abs(float(x[0][x[1]])));return dict(row=int(point['row']),field=field,value=float(point[field]),xyz=[float(point[k]) for k in ['x','y','z']],passes_unchanged_1e_minus_7_gate=abs(float(point[field]))<=1e-7)
 def component_max(field):
  x=max(evolved,key=lambda row:abs(float(row[field])));return {k:(float(v) if k not in ['state','row','face','j','k'] else int(v)) for k,v in x.items()}
 pointwise[f.stem]=dict(initial=maximum(initial),evolved=maximum(evolved),evolved_Axx_max=component_max('Axx'),evolved_LambdaX_max=component_max('LambdaX'),all_rows=len(evolved),no_exclusions=True)
 target=out/(f.name+'.gz')
 with gzip.open(target,'wb') as g:g.write(f.read_bytes())
 assert gzip.open(target,'rb').read()==f.read_bytes();compressed.append(dict(input=str(f),input_sha256=sha(f),gzip=target.name,gzip_sha256=sha(target),lossless=True))
project=r.resolve();deps={}
for f in src.glob('*-deps.stdout'):
 paths=f.read_text().replace('\\\n',' ').split();live=[x for x in paths if x.startswith(str(project/'src')+'/') or x.startswith(str(project/'tst')+'/')];deps[f.name]=dict(live_project_headers=live,sha256=sha(f))
inputs={}
for command in manifest['runs'].values():
 for name in command['command']:
  q=Path(name)
  if q.is_file() and q.suffix in ['.bin','.checkpoint','.dat']:inputs[str(q)]=sha(q)
results=dict(scope='Frozen instantaneous normal-jet diagnostics only; no solve/evolution/engine change',scientific_gate=1e-7,all_candidate_evolved_gates_fail=True,derivative_basis='all coefficients free in r, (R^2-r^2)/(2R), or -log(r/R)',orders=orders,negative_guards=failures,pointwise=pointwise,runs=data,compressed_rows=compressed,input_hashes=inputs,compiled_source_sha256=sha(src/'compiled-source.tar.gz'),dependency_proof=deps,source_inputs_unchanged=manifest['inputs_unchanged'],runtime_unchanged=manifest['runtime_unchanged'],limitations=['No complete frozen wormhole regular-limit rate because fresh solved beta first/second jets are absent','No nonlinear evolution with new radial derivatives','Coarse full angular+radial first derivative rates do not establish global fourth order','Cross-layer plan requires aligned active angular ray lattices; no shell AMR claim','Initial exact radial series is only an oracle, never a coefficient constraint'])
(out/'analysis.json').write_text(json.dumps(results,indent=2)+'\n')
fig,ax=plt.subplots(1,2,figsize=(11.5,4.1));names=['pole-v16-7-0','pole-v16-7-2','pole-v16-9-0','pole-v16-9-2','pole-v16-LS-13-2','pole-v16-LS-15-2'];labels=['r:7','log:7','r:9','log:9','log LS6:13','log LS6:15'];x=range(len(names));ax[0].semilogy(x,[data[n]['initial_max_pole'] for n in names],'o-',label='Exact initial data, discretized jets');ax[0].semilogy(x,[data[n]['evolved_max_pole'] for n in names],'s-',label='Frozen evolved midpoint');ax[0].axhline(1e-7,color='black',ls='--',label='Unchanged pole target');ax[0].set_xticks(list(x));ax[0].set_xticklabels(labels,rotation=25);ax[0].set_ylabel('Max |tensor/gauge pole numerator|');ax[0].legend(fontsize=8);ax[0].set_title('Initial improvement does not imply evolved compatibility')
for item in orders:
 ax[1].loglog(item['n'],item['radial_AD_second_max'],'o-' if item['nodes']==7 else 's--',label=f"{item['layers']} layers, {item['nodes']} nodes")
ax[1].set_xlabel('Active cells per block direction N');ax[1].set_ylabel('All-node max second-normal AD error');ax[1].set_title('Normal-jet consistency, all surfaces retained');ax[1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'normal-error-budget.pdf');fig.savefig(out/'normal-error-budget.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(11.5,4.3));item=pointwise['pole-v16-7-2'];xx=item['evolved_Axx_max'];keys=['baseline','chi_normal','chi_tangent','metric_normal','metric_tangent','independent_lambda','curvature'];ax[0].bar(range(len(keys)),[xx['Axx_'+k] for k in keys]);ax[0].set_xticks(range(len(keys)));ax[0].set_xticklabels(keys,rotation=50,ha='right');ax[0].set_title(f"log7 Axx maximum, row {xx['row']}");ax[0].set_ylabel('Additive pole numerator contribution');ll=item['evolved_LambdaX_max'];keys=['metric_normal','metric_tangent','independent_lambda','trace_normal','trace_tangent','curvature'];ax[1].bar(range(len(keys)),[ll['Lambda0_'+k] for k in keys]);ax[1].set_xticks(range(len(keys)));ax[1].set_xticklabels(keys,rotation=50,ha='right');ax[1].set_title(f"log7 LambdaX maximum, row {ll['row']}");fig.tight_layout();fig.savefig(out/'evolved-pole-decomposition.pdf');fig.savefig(out/'evolved-pole-decomposition.png',dpi=160);plt.close(fig)
print(json.dumps(dict(mms_cases=len([x for x in data if x.startswith('mms-')]),guard_rejections=len(failures),all_evolved_candidates_fail=True,live_project_headers=sum(len(x['live_project_headers']) for x in deps.values()),source_inputs_unchanged=manifest['inputs_unchanged'],runtime_unchanged=manifest['runtime_unchanged'])))
