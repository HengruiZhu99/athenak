#!/usr/bin/env python3
"""Native TT continuation histories, radial profiles and complete geometric slice."""
from pathlib import Path
import argparse,json,ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();R=a.root.resolve();E=R/'docs/validation/goal1-minkowski-20261018';P=E/'native-tensor-propagation-v1';W=R/'build-goal1-minkowski-20261018/native-tensor-propagation-v1';old=json.loads((E/'native-tensor-pilot-v1/resolved-entry-v3-review.json').read_text());L=json.loads((P/'bounded-propagation-protocol-v7.json').read_text())['length'];A=json.loads((E/'native-tensor-pilot-v1/primary-normalization-v2.json').read_text())['full_configuration_scale'];scale=np.full(22,A/L);config=[0,1,2,3,4,5,6,18,19,20,21];scale[config]=A;src=Path(__file__).with_name('review_goal1_clock_pilots.py');ns={'np':np};tree=ast.parse(src.read_text());exec(compile(ast.Module(body=[f for f in tree.body if isinstance(f,ast.FunctionDef)and f.name=='reference_weights'],type_ignores=[]),str(src),'exec'),ns)
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);profile,paxes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);slices,saxes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
for ax,pax,sax,sector in zip(axes,paxes,saxes,['plus','cross']):
 hist=[old['cases'][f'tt-{sector}-resolved4-v3']['initial'],old['cases'][f'tt-{sector}-resolved4-v3']['final']]
 for report,label in [(f'{sector}-segment1-review.json',f'tt-{sector}-to005-segment1'),(f'{sector}-final-review.json',f'tt-{sector}-to005-segment2')]:hist.append(json.loads((P/report).read_text())['cases'][label]['final'])
 for k,color in [('fields','#2066a0'),('first','#d34c27'),('configuration_second','#2d8b57')]:
  data=[(h['time'],h['regions']['all']['errors']['fixed_reference_Penrose'][k]['rms'])for h in hist if k!='fields'or h['time']>0];ax.plot([x[0]for x in data],[x[1]for x in data],'-o',color=color,label=k.replace('configuration_second','config Hessian'))
 ax.plot([h['time']for h in hist],[h['regions']['all']['constraints']['fixed_reference_Penrose']['dimensionless']['H']['rms']for h in hist],':s',color='#633c9e',label='H constraint');ax.axhline(.01,color='black',linestyle='--',linewidth=.7,label='RMS budget');ax.set_yscale('log');ax.set_ylim(1e-5,.02);ax.set_xlabel('Native time');ax.set_ylabel('Fixed-v3 normalized RMS');ax.set_title(sector+' polarization');ax.grid(alpha=.2)
 label=f'tt-{sector}-to005-segment2';case=W/label
 with(case/'cubed_sphere_native_last_jets.bin').open('rb')as f:n,c=np.fromfile(f,np.uint64,2)
 x=np.memmap(case/'cubed_sphere_native_last_jets.bin',dtype='<f8',mode='r',offset=16,shape=(int(n),300));target=np.memmap(W/('analysis-'+label)/'final-target.bin',dtype='<f8',mode='r',offset=16,shape=(int(n),294));delta=(x[:,14:].reshape(-1,22,13)-target[:,:286].reshape(-1,22,13))/scale[None,:,None];first=L*np.sqrt(np.sum(delta[:,:,1:4]**2,axis=(1,2)));hess=L*L*np.sqrt(np.sum(delta[:,config,4:]**2,axis=(1,2)));radius=np.sqrt(np.sum(x[:,:3]**2,axis=1));weight=ns['reference_weights'](x)['fixed_reference_Penrose'];bins=np.linspace(0,.7,36)
 for z,name,color in [(first,'first','#d34c27'),(hess,'config Hessian','#2d8b57')]:
  xx=[];yy=[]
  for lo,hi in zip(bins[:-1],bins[1:]):
   mask=(radius>=lo)&(radius<hi)
   if mask.any():xx.append((lo+hi)/2);yy.append(np.sqrt(np.sum(weight[mask]*z[mask]**2)/weight[mask].sum()))
  pax.plot(xx,yy,color=color,label=name)
 pax.set_yscale('log');pax.set_ylim(1e-12,.1);pax.set_xlabel('Compact radius');pax.set_ylabel('Reference-volume radial RMS');pax.set_title(sector+' at t=.005');pax.grid(alpha=.2)
 mask=abs(x[:,1])<.035;artist=sax.scatter(x[mask,0],x[mask,2],c=np.maximum(first[mask],1e-12),s=2,norm=LogNorm(1e-12,.05),cmap='viridis',rasterized=True);sax.set_aspect('equal');sax.set_xlabel('x');sax.set_ylabel('z');sax.set_title(sector+' first error, |y|<.035')
axes[0].legend(fontsize=8);fig.suptitle('Actual native coupled TT: bounded propagation through .005, not crossing acceptance',fontsize=11);fig.savefig(P/'native-TT-propagation-histories.png',dpi=180);plt.close(fig);paxes[0].legend();profile.suptitle('All native nodes in each radial bin; fixed v3 normalization');profile.savefig(P/'native-TT-propagation-radial.png',dpi=180);plt.close(profile);slices.colorbar(artist,ax=saxes,label='Normalized local first error (display floor 1e-12)');slices.savefig(P/'native-TT-propagation-slices.png',dpi=180);plt.close(slices)
