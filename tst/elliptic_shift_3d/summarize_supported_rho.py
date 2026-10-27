#!/usr/bin/env python3
"""Preserve unsupported geometries while reducing absolute weighted MMS errors."""
import argparse,collections,hashlib,json,math,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,default=Path('docs/elliptic-shift-3d/evidence'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
paths={k:a.evidence/f'supported-rho-log-{k}-20261027'/'results.json' for k in ['mms','actual','source-noise']};data={k:json.loads(p.read_text()) for k,p in paths.items()};mms=data['mms'];keys=['family','translated','h','power','fractional_exponent'];groups=collections.defaultdict(lambda:collections.defaultdict(list))
for row in mms:
 if row['phase']=='mms':groups[tuple(row[k] for k in keys)][row['basis']].append(row)
summary={'geometry_summary':mms[-1],'actual':data['actual'][-1],'source_noise':data['source-noise'][-1],'groups':[],'unsupported':[],'sources':[]}
for k,p in paths.items():summary['sources'].append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
for key,bases in sorted(groups.items()):
 entry=dict(zip(keys,key));entry['basis']={}
 for basis in ['base125','supported_rho','base_matched_support']:
  rows=bases[basis];weight=sum(r['physical_query_volume'] for r in rows);maximum=max((r['physical_H_error'] for r in rows),default=None);rms=math.sqrt(sum(r['physical_query_volume']*r['physical_H_error']**2 for r in rows)/weight) if rows else None
  complete=len(rows)==8;entry['basis'][basis]={'query_count':len(rows),'complete_8_queries':complete,'maximum':maximum if complete else None,'weighted_rms':rms if complete else None,'supported_subset_maximum':maximum,'supported_subset_weighted_rms':rms,'missing_octants':sorted(set(range(8))-{r['octant'] for r in rows})}
 summary['groups'].append(entry)
for row in mms:
 if row['phase']=='unsupported':
  attempts=[r for r in mms if r['phase']=='support_attempt' and all(r[k]==row[k] for k in ['translated','octant','h'])];summary['unsupported'].append({'geometry':row,'all_attempts':attempts})
summary['origin_comparisons']=[]
for family in ['wormhole','angular_xy','angular_xz','angular_zz','radial_fractional','exact_rho_polynomial']:
 exponents=[0] if family in ['wormhole','exact_rho_polynomial'] else [.3,1.1,2.4]
 powers=[4,2,1.09,.5] if family=='exact_rho_polynomial' else [4 if family=='wormhole' else 2]
 for exponent in exponents:
  for power in powers:
   rows=[r for r in summary['groups'] if r['family']==family and r['translated']==0 and r['power']==power and r['fractional_exponent']==exponent];rows.sort(key=lambda r:-r['h'])
   if not rows:continue
   item={'family':family,'power':power,'fractional_exponent':exponent,'cases':rows,'orders':{}}
   for basis in ['base125','supported_rho','base_matched_support']:
    errors=[r['basis'][basis]['maximum'] for r in rows];item['orders'][basis]=[math.log(x/y,2) for x,y in zip(errors,errors[1:])] if all(x and x>0 for x in errors) else None
   summary['origin_comparisons'].append(item)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
os.environ.setdefault('MPLCONFIGDIR','/tmp/codex-supported-rho-matplotlib');os.environ.setdefault('XDG_CACHE_HOME','/tmp/codex-supported-rho-fontcache');Path(os.environ['XDG_CACHE_HOME']).mkdir(exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FixedFormatter,NullFormatter
fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained');labels={'base125':'base125','supported_rho':'supported rho','base_matched_support':'matched-support base'}
for basis in labels:
 row=next(r for r in summary['origin_comparisons'] if r['family']=='wormhole');ax[0].loglog([r['h'] for r in row['cases']],[r['basis'][basis]['maximum'] for r in row['cases']],'o-',label=labels[basis])
ax[0].set_title('Initial radial wormhole: origin 8/8');ax[0].set_ylabel('Maximum physical H error')
for basis in ['base125','supported_rho']:
 for power in [4,1.09,.5]:
  row=next(r for r in summary['origin_comparisons'] if r['family']=='exact_rho_polynomial' and r['power']==power);ax[1].loglog([r['h'] for r in row['cases']],[r['basis'][basis]['maximum'] for r in row['cases']],label=f'{labels[basis]}, p={power}',linestyle='-' if basis=='supported_rho' else '--')
ax[1].set_title('p log(rho) + rho + polynomial');ax[1].set_ylabel('Maximum physical H error')
for exponent in [.3,1.1,2.4]:
 for basis in ['base125','supported_rho','base_matched_support']:
  rows=[r for r in summary['origin_comparisons'] if r['family'].startswith('angular_') and r['fractional_exponent']==exponent];heights=[max(r['cases'][i]['basis'][basis]['weighted_rms'] for r in rows) for i in range(4)];ax[2].loglog([.04,.02,.01,.005],heights,label=f'{labels[basis]}, s={exponent}',linestyle='-' if basis=='supported_rho' else '--' if basis=='base125' else ':')
ax[2].set_title('3 angular modes: origin 8/8 only');ax[2].set_ylabel('Largest physical-volume query RMS')
for axis in ax:axis.set_xlabel('h (5 translated cases remain unsupported)');axis.grid(True,which='both',alpha=.25);axis.legend(fontsize=6.5);axis.xaxis.set_major_locator(FixedLocator([.005,.01,.02,.04]));axis.xaxis.set_major_formatter(FixedFormatter(['.005','.01','.02','.04']));axis.xaxis.set_minor_formatter(NullFormatter())
fig.savefig(out/'supported-rho-controls.png',dpi=160)
print(json.dumps({'geometry_summary':summary['geometry_summary'],'unsupported_geometry_count':len(summary['unsupported']),'complete_origin_cases':len(summary['origin_comparisons']),'full_norms_in_incomplete_groups':None},indent=2))
