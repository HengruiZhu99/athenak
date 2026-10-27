#!/usr/bin/env python3
"""Reduce frozen falsifiers without excluding unsupported or fractional cases."""
import argparse,hashlib,json,math,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,default=Path('docs/elliptic-shift-3d/evidence'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
names=['rho-log-mms-20261027','rho-log-raw-falsifier-20261027','rho-log-actual-20261027','rho-log-source-noise-20261027','bounded-rho-log-mms-20261027','bounded-rho-log-actual-20261027','bounded-rho-log-source-noise-20261027']
data={name:json.loads((a.evidence/name/'results.json').read_text()) for name in names};mms=data['bounded-rho-log-mms-20261027'];bases=['base','rho_raw','bounded_blend','fullodd'];summary={'sources':[], 'mms_builds':{},'wormhole':[],'angular':[],'power_rho':[],'actual':data['bounded-rho-log-actual-20261027'][-1],'source_noise':data['bounded-rho-log-source-noise-20261027'][-1]}
for name in names:
 for filename in ['manifest.json','results.json']:
  f=a.evidence/name/filename;summary['sources'].append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
for basis in [*bases,'rho']:
 rows=[r for r in mms if r['phase']=='build' and r['basis']==basis];status={}
 for r in rows:status[r['status']]=status.get(r['status'],0)+1
 cases=[r for r in mms if r['phase']=='mms' and r['basis']==basis]
 summary['mms_builds'][basis]={'statuses':status,'maximum_condition':max(r['condition'] for r in rows),'maximum_gradient_l1':max(r['scaled_gradient_l1'] for r in rows),'maximum_hessian_l1':max(r['scaled_hessian_l1'] for r in rows),'maximum_scaled_noise':max(r['scaled_noisy_derivative_change'] for r in cases)}
def errors(family,basis,**filters):
 return [r for r in mms if r['phase']=='mms' and r['family']==family and r['basis']==basis and all(r[k]==v for k,v in filters.items())]
def series(family,basis,**filters):
 rows=[{'h':h,'maximum_H_error':max(r['physical_H_error'] for r in errors(family,basis,h=h,**filters))} for h in [.04,.02,.01,.005]]
 return {'basis':basis,'cases':rows,'orders':[math.log(x['maximum_H_error']/y['maximum_H_error'],2) for x,y in zip(rows,rows[1:])]}
for basis in bases:summary['wormhole'].append(series('wormhole',basis,translated=0))
for family in ['angular_xy','angular_xz','angular_zz','radial_fractional']:
 for exponent in [.3,1.1,2.4]:
  for translated in [0,1]:
   baseline=series(family,'base',fractional_exponent=exponent,translated=translated)
   for basis in bases:
    s=series(family,basis,fractional_exponent=exponent,translated=translated);s.update(family=family,exponent=exponent,translated=translated)
    s['maximum_error_ratios_to_base']=[x['maximum_H_error']/y['maximum_H_error'] for x,y in zip(s['cases'],baseline['cases'])];summary['angular'].append(s)
for power in [4,2,1.09,.5]:
 for translated in [0,1]:
  for basis in bases:
   s=series('exact_rho_polynomial',basis,power=power,translated=translated);s.update(power=power,translated=translated);summary['power_rho'].append(s)
theta=[r for r in mms if r['phase']=='bounded_theta'];summary['mms_theta']={'rows':len(theta),'mixed':sum(r['theta']<1 for r in theta),'minimum':min(r['theta'] for r in theta),'maximum':max(r['theta'] for r in theta)}
paired=[]
for family in ['angular_xy','angular_xz','angular_zz']:
 baseline={(r['translated'],r['octant'],r['h'],r['fractional_exponent']):r for r in errors(family,'base')}
 for r in errors(family,'bounded_blend'):
  b=baseline[(r['translated'],r['octant'],r['h'],r['fractional_exponent'])];paired.append({'family':family,'translated':r['translated'],'octant':r['octant'],'h':r['h'],'exponent':r['fractional_exponent'],'base_H_error':b['physical_H_error'],'bounded_H_error':r['physical_H_error'],'ratio':r['physical_H_error']/b['physical_H_error']})
summary['worst_paired_angular_ratio']=max(paired,key=lambda r:r['ratio']);summary['worst_paired_angular_absolute_increase']=max(paired,key=lambda r:r['bounded_H_error']-r['base_H_error'])
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
os.environ.setdefault('MPLCONFIGDIR','/tmp/codex-rho-log-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(13,3.9),layout='constrained');labels={'base':'base log','rho_raw':'raw +rho','bounded_blend':'bounded blend','fullodd':'full odd'}
for row in summary['wormhole']:axes[0].loglog([x['h'] for x in row['cases']],[x['maximum_H_error'] for x in row['cases']],'o-',label=labels[row['basis']])
axes[0].set_title('Initial radial wormhole: 8-point maximum');axes[0].set_xlabel('Cartesian spacing h');axes[0].set_ylabel('Physical H error')
for basis in ['base','bounded_blend','rho_raw']:
 for power in [4,1.09,.5]:
  row=next(r for r in summary['power_rho'] if r['basis']==basis and r['power']==power and r['translated']==0)
  axes[1].loglog([x['h'] for x in row['cases']],[x['maximum_H_error'] for x in row['cases']],label=f'{labels[basis]}, p={power}',linestyle='-' if basis=='bounded_blend' else '--' if basis=='base' else ':')
axes[1].set_title('p log(rho) + rho + polynomial');axes[1].set_xlabel('Cartesian spacing h');axes[1].set_ylabel('Physical H error (all powers retained)')
for basis in ['rho_raw','bounded_blend','fullodd']:
 rows=[r for r in summary['angular'] if r['basis']==basis and r['family'].startswith('angular_')]
 axes[2].plot([.04,.02,.01,.005],[max(r['maximum_error_ratios_to_base'][i] for r in rows) for i in range(4)],'o-',label=labels[basis])
axes[2].axhline(1,color='.5',linestyle=':');axes[2].set_xscale('log');axes[2].set_yscale('log');axes[2].set_title('Worst angular maximum-error ratio');axes[2].set_xlabel('Cartesian spacing h');axes[2].set_ylabel('Candidate / base (3 modes, 3 exponents, 2 offsets)')
for ax in axes:ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=6.6)
fig.savefig(out/'rho-log-tradeoff.png',dpi=160)
print(json.dumps({'mms_builds':summary['mms_builds'],'mms_theta':summary['mms_theta'],'worst_paired_angular_ratio':summary['worst_paired_angular_ratio'],'bounded_power_rho':[r for r in summary['power_rho'] if r['basis']=='bounded_blend' and r['translated']==0]},indent=2))
