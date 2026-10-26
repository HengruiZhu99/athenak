#!/usr/bin/env python3
"""Selected native closure comparisons; full failures remain in JSON evidence."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p = argparse.ArgumentParser()
p.add_argument('review',type=Path)
p.add_argument('field_time',type=Path)
p.add_argument('output',type=Path)
a = p.parse_args()
j = json.loads(a.review.read_text());assert j['common_target_reached']
checks = {x['region_field']:x for x in j['constraint_checks']}
time = json.loads(a.field_time.read_text())
fig,axes = plt.subplots(2,3,figsize=(12,7),layout='constrained')
for col,field in enumerate(['H_l2','M_conformal_l2','Z_conformal_l2']):
    x = checks['shell/fixed_outer_collar/'+field];ax=axes[0,col]
    for name in ['quintic','interior']:ax.semilogy([16,24,32],x[name],'o-',label=name)
    ax.set(title='Outer collar '+field[0]+' RMS',xticks=[16,24,32],xlabel='Balanced bulk N')
    ax.grid(alpha=.2);ax.legend()
for col,(component,name) in enumerate([(14,'Lambda_x'),(15,'Lambda_y'),(17,'Theta')]):
    x = next(x for x in j['field_checks'] if x['region']=='shell/fixed_outer_collar'
             and x['component']==component and x['norm']=='linf')
    ax=axes[1,col];ax.semilogy([16,24,32],x['values'],'o-',color='tab:purple',label='Closure difference')
    ax.semilogy([24],[time['regions']['shell/fixed_outer_collar']['components'][component]['linf']],
                'x',color='gray',markersize=8,label='N24 half-CFL difference')
    ax.set(title=name+' maximum pointwise difference',xticks=[16,24,32],xlabel='Balanced bulk N')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Native20% angular pulse t=0.2, matched Rb=0.98 grids, immutable Minkowski reference\n'
             'Constraints refine within both closures; full closure consistency fails. '
             'Alternative temporal/background controls pending.')
fig.savefig(a.output,dpi=170);fig.savefig(a.output.with_suffix('.pdf'))
