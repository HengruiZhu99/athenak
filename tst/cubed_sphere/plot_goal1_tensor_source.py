#!/usr/bin/env python3
"""Plot saved native TT source conditioning and late coupled clock errors."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('tensor',type=Path);p.add_argument('late',type=Path);a=p.parse_args();review=json.loads((a.tensor/'scri-approach-review.json').read_text());fig,ax=plt.subplots(figsize=(8.5,5.5),layout='constrained')
for sector,color in [('plus','#235b8e'),('cross','#b35136')]:
 rows=[r for r in review['rows']if r['case'].startswith(sector+'-')];ax.loglog([r['Omega']for r in rows],[r['scaled_source_error']for r in rows],'o-',label=sector+' unfactored finite source',color=color)
x=np.geomspace(2e-8,2e-4,100);ax.loglog(x,.5*np.finfo(float).eps/x,'--',color='0.5',label='0.5 epsilon64 / Omega guide');ax.axhline(1e-10,color='#9f6d20',ls=':',label='raw comparison target (two exceedances)');ax.axhline(review['known_exact_face_source_error'],color='#548049',lw=1.5,label='compatible exact-face quotient');ax.invert_xaxis();ax.set(xlabel='Finite Omega, approaching scri to the right',ylabel='Maximum scaled native22 source error',title='Both native TT polarizations: cancellation in finite-Omega division');ax.legend(fontsize=8);ax.grid(alpha=.2,which='both');fig.supxlabel('Continuum point evaluation only; no numerical endpoint stability or tensor propagation admission.',fontsize=9);fig.savefig(a.tensor/'source-conditioning.png',dpi=160);fig.savefig(a.tensor/'source-conditioning.svg');plt.close(fig)
segments=json.loads((a.late/'inspected-segments.json').read_text())['segments'];fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained');times=[r['time']for r in segments]
for k,color,label in [('fields','#235b8e','native fields'),('first','#b35136','first derivatives'),('configuration_second','#548049','configuration Hessians')]:axes[0].semilogy(times,[r['max_regional_reference_RMS'][k]for r in segments],'o-',color=color,label=label)
axes[1].semilogy(times,[r['max_regional_constraint_RMS']for r in segments],'o-',color='#6f4d86',label='max over nine regions and H/M/Z/Theta')
for axis in axes:axis.axhline(.01,color='0.5',ls=':',label='declared short-pilot RMS budget');axis.set(xlabel='Coordinate time',ylabel='Maximum regional normalized RMS');axis.grid(alpha=.2);axis.legend(fontsize=8)
axes[0].set_title('Full native coupled clock: field/derivative errors');axes[1].set_title('Physical and conformal constraints');fig.suptitle('Compatible continuation through t=.4, unchanged native second-order method',fontsize=12);fig.supxlabel('Measured errors pass short budgets; causal boundaries, long-time/timestep and exact-scri numerical gates remain open.',fontsize=9);fig.savefig(a.late/'native-clock-late-errors.png',dpi=160);fig.savefig(a.late/'native-clock-late-errors.svg');plt.close(fig)
receipt={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inputs':{str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in [a.tensor/'scri-approach-review.json',a.late/'inspected-segments.json']},'scope':'Saved continuum TT conditioning and actual native coupled clock diagnostics, distinct scopes.'};(a.tensor/'plot-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
