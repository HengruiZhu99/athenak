#!/usr/bin/env python3
"""Build standalone illustrated LaTeX from preserved scalar evidence; no runs."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'docs/validation/test-scalar-20261016'
sys.path.insert(0,str(ROOT/'tst/test_scalar'))
from scalar_evidence import waveform
INPUTS={}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(name):
 p=E/name;INPUTS[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())
def coords(x,y):return ' '.join(f'({float(a):.12g},{float(b):.12g})' for a,b in zip(x,y))
def plot(x,y,style,label=None):
 text=r'\addplot['+style+'] coordinates {'+coords(x,y)+'};\n'
 if label:text+=r'\addlegendentry{'+label+'}\n'
 return text
def axis(body,options):return '\\begin{axis}['+options+']\n'+body+'\\end{axis}\n'
def pair(left,right):return '\\begin{center}\\begin{tikzpicture}\n'+left+right+'\\end{tikzpicture}\\end{center}\n'
def texnum(value):
 exponent=int(math.floor(math.log10(abs(value))));return f'{value/10**exponent:.4g}'+r'\times10^{'+str(exponent)+'}'
def smooth(z):
 if z<=0:return 0.
 if z>=1:return 1.
 a=math.exp(-1/z);b=math.exp(-1/(1-z));return a/(a+b)
def mapfig():
 rho=np.linspace(0,24,241);o=np.array([1-smooth((r-8)/8)*r/24 for r in rho]);z=np.clip((rho-8)/8,0,1);o0=1-z**3*(10-15*z+6*z*z)*rho/24
 hp=[];h0=[]
 for q,v,v0 in zip(rho,o,o0):
  if q==24:hp.append(1.);h0.append(1.);continue
  r=q/v;r0=q/v0;hp.append(smooth((r-8)/8)*(r/math.sqrt(r*r+100)+3/r) if r else 0.);h0.append(r0/math.sqrt(r0*r0+100))
 left=axis(plot(rho,o,'blue,thick','Physical $C^\\infty$')+plot(rho,o0,'bad,dashed,thick','Quintic reference'),r'width=8cm,height=5.8cm,xmin=0,xmax=24,xtick={0,8,16,24},ymin=0,ymax=1.05,xlabel={$\rho$},ylabel={$\Omega$},title={Separate fixed conformal factors},legend pos=south west')
 right=axis(plot(rho,hp,'blue,thick',r'Schwarzschild $h^\prime$')+plot(rho,h0,'bad,dashed,thick',r'Reference $h_0^\prime$'),r'at={(8.5cm,0)},anchor=south west,width=8cm,height=5.8cm,xmin=0,xmax=24,xtick={0,8,16,24},ymin=0,ymax=1.3,xlabel={$\rho$},ylabel={Height derivative $h^\prime(r)$},title={Different inner and asymptotic slicings},legend pos=north west')
 return pair(left,right)
def domainfig():
 def path(points):return ' -- '.join(f'({x:.5f},{y:.5f})' for x,y in points)
 def point(p,s,u):
  f=(1-math.exp(2*s))/(1-math.exp(2));d=(1-f)*3+f*24/math.sqrt(1+u*u)
  return ((-1 if p==1 else 1)*d*.127,d*u*.127) if p<2 else (d*u*.127,(-1 if p==3 else 1)*d*.127)
 lines=[r'\begin{center}\begin{tikzpicture}[x=1.4cm,y=1.4cm,font=\scriptsize]',r'\node[font=\small\bfseries] at (0,3.5) {Equatorial cut: $\rho\leq24$};']
 for p,color in enumerate(['blue','blue','good','good']):
  boundary=[point(p,0,u) for u in np.linspace(-1,1,21)]+[point(p,1,u) for u in np.linspace(1,-1,41)]
  lines.append(r'\fill['+color+'!7] '+path(boundary)+' -- cycle;')
  for s in np.linspace(0,1,7):lines.append(r'\draw[black!32,line width=.25pt] '+path([point(p,s,u) for u in np.linspace(-1,1,51)])+';')
  for u in np.linspace(-1,1,9):lines.append(r'\draw[black!32,line width=.25pt] '+path([point(p,s,u) for s in np.linspace(0,1,31)])+';')
 for v in np.linspace(-3,3,13):
  lines.append(f'\\draw[blue!35,line width=.2pt] ({v*.127:.5f},-.381)--({v*.127:.5f},.381);')
  lines.append(f'\\draw[blue!35,line width=.2pt] (-.381,{v*.127:.5f})--(.381,{v*.127:.5f});')
 lines.extend([r'\draw[blue,thick] (-.381,-.381) rectangle (.381,.381);',r'\draw[black,thick] (0,0) circle (3.048);',r'\node[fill=white,inner sep=1pt] at (2.0,0) {$+X$};',r'\node[fill=white,inner sep=1pt] at (-2.0,0) {$-X$};',r'\node[fill=white,inner sep=1pt] at (0,2.0) {$+Y$};',r'\node[fill=white,inner sep=1pt] at (0,-2.0) {$-Y$};',r'\node[align=center,fill=white,inner sep=1pt] at (-1.3,-.72) {Cartesian bulk\\$L=3$};',r'\draw[-{Stealth}] (-.85,-.65)--(-.25,-.25);',r'\node[fill=white,inner sep=1pt] at (0,-3.18) {Exact scri: $\Omega=0$};',r'\node[font=\small\bfseries] at (5.05,3.5) {Six shell faces};'])
 az=math.radians(35);el=math.radians(25)
 def project(x):
  return (5.05+1.43*(-math.sin(az)*x[0]+math.cos(az)*x[1]),1.76+1.43*(-math.sin(el)*math.cos(az)*x[0]-math.sin(el)*math.sin(az)*x[1]+math.cos(el)*x[2]))
 for face in [1,3,5,0,2,4]:
  a=face//2;b=(a+1)%3;c=(a+2)%3;sign=-1 if face%2 else 1;color={0:'blue',2:'good',4:'purple'}.get(face,'black!22');style=color+(',dashed' if face%2 else ',line width=.45pt')
  for swap in [False,True]:
   for fixed in [-1,-.5,0,.5,1]:
    pts=[]
    for variable in np.linspace(-1,1,51):
     u,v=(fixed,variable) if swap else (variable,fixed);d=1/math.sqrt(1+u*u+v*v);xyz=[0.,0.,0.];xyz[a]=sign*d;xyz[b]=d*u;xyz[c]=d*v;pts.append(project(xyz))
    lines.append('\\draw['+style+'] '+path(pts)+';')
 for face,name,color in [(0,'+X','blue'),(2,'+Y','good'),(4,'+Z','purple')]:
  x=[0.,0.,0.];x[face//2]=1;xx,yy=project(x);lines.append(f'\\node[fill=white,inner sep=.7pt,text={color}] at ({xx:.5f},{yy:.5f}) {{$'+name+'$};')
 lines.extend([r'\node[align=center] at (5.05,.02) {Back faces $-X,-Y,-Z$ dashed\\No polar coordinate singularity};',r'\node[font=\small\bfseries] at (5.05,-.64) {Schwarzschild inset};',r'\draw[black!20] (3.75,-3.18) rectangle (6.5,-.86);',r'\draw[bad,thick] (5.05,-2) circle (.9);',r'\fill[black!22] (4.825,-2.225) rectangle (5.275,-1.775);',r'\draw[black,thick] (4.825,-2.225) rectangle (5.275,-1.775);',r'\draw[blue,dashed] (4.88125,-2.16875) rectangle (5.21875,-1.83125);',r'\draw[-{Stealth},thick,good] (4.42,-2)--(4.78,-2);',r'\draw[-{Stealth},thick,good] (5.68,-2)--(5.32,-2);',r'\node[fill=white,inner sep=1pt] at (5.05,-1.13) {$r_H=M$};',r'\node[align=center,fill=white,inner sep=1pt] at (5.05,-2.64) {Removed cube: $d=.25M$\\Dashed control: $d=.1875M$};',r'\node at (5.05,-3.01) {Arrows leave the evolved domain};',r'\end{tikzpicture}\end{center}'])
 return '\n'.join(lines)
def minkfig(m):
 fields=[('psi',r'$\psi$','blue,mark=*'),('v',r'$v_{\rm rec}$','bad,mark=square*'),('Dpsi',r'$D\psi$','good,mark=triangle*'),('reduction',r'$C$','purple,mark=diamond*')];axes=[]
 for i,norm in enumerate(['L2','Linf']):
  body=''
  for field,label,style in fields:
   q=next(c for c in m['checks'] if c['time']==12 and c['region']=='all' and c['field']==field and c['norm']==norm);body+=plot(m['resolutions'],q['errors'],style+',thick',label)
  opts=(r'at={(8.5cm,0)},anchor=south west,' if i else '')+r'width=8cm,height=6.4cm,xmode=log,ymode=log,log ticks with fixed point,xtick={24,32,40},xlabel={$N$},ylabel={Error / residual},legend pos=north east,title={' + ('Volume RMS' if i==0 else 'Maximum norm')+'}'
  axes.append(axis(body,opts))
 return pair(*axes)
def schfailfig(s):
 ns=s['resolutions'];error=[r for r in s['records'] if r['time']==100][0]['errors'];e=[v['psi']['L2'] for v in error]
 body=plot(ns,e,'blue,thick,mark=*','Measured $E_2(\\psi)$')+plot([48,60],[e[1],e[1]*(48/60)**1.5],'bad,dashed,thick',r'$p=1.5$ from $N48$')+plot([60],[e[2]],'bad,only marks,mark=x,mark size=4pt')
 left=axis(body,r'width=8cm,height=6.1cm,xmode=log,ymode=log,log ticks with fixed point,xtick={36,48,60},xlabel={$N$},ylabel={$E_2(\psi)$},title={Late field: $t=100M$},legend pos=north east')
 times=[2,6,12,100];body=''
 for field,label,style in [('psi',r'$\psi$','blue,mark=*'),('v',r'$v$','bad,mark=square*'),('Dpsi',r'$D\psi$','good,mark=triangle*'),('reduction',r'$C$','purple,mark=diamond*')]:
  values=[min(c['orders'][1] for c in s['field_checks'] if c['time']==t and c['field']==field) for t in times];body+=plot(times,values,style+',thick',label)
 body+=plot([2,100],[1.5,1.5],'black,dashed',r'Required $1.5$')
 right=axis(body,r'at={(8.5cm,0)},anchor=south west,width=8cm,height=6.1cm,xmode=log,xtick={2,6,12,100},log ticks with fixed point,xlabel={$t/M$},ylabel={Minimum of both norm orders},title={$N48/N60$ pair},legend style={at={(.5,-.32)},anchor=north,legend columns=3}')
 return pair(left,right)
def qnmfig(s,q):
 target=q['expected'];x=target['Momega_real'];y=-target['Momega_imag'];body=f'\\path[fill=good!8,draw=good!50,dashed] (axis cs:{x*.98},{y*.95}) rectangle (axis cs:{x*1.02},{y*1.05});\n'
 for records,label,style in [(q['records'],'Independent 4800','blue,mark=*'),(s['QNM'][-1]['fits'],'3D $N60$','bad,dashed,mark=square,mark size=3pt')]:body+=plot([r['omega_real'] for r in records],[r['decay'] for r in records],style+',thick',label)
 body+=plot([x],[y],'black,only marks,mark=+,mark size=5pt','Scalar target')
 return '\\begin{center}\\begin{tikzpicture}\n'+axis(body,r'width=12cm,height=6.7cm,xlabel={$M\omega_R$},ylabel={$M\Gamma$},xmin=.471,xmax=.496,ymin=.091,ymax=.102,scaled ticks=false,xtick={.475,.480,.485,.490,.495},ytick={.092,.094,.096,.098,.100,.102},xticklabel style={/pgf/number format/fixed,/pgf/number format/precision=3,/pgf/number format/fixed zerofill},yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=3,/pgf/number format/fixed zerofill},legend pos=north west')+'\\end{tikzpicture}\\end{center}'
def partialfig(r):
 base=E/'aurora/cpu-schwarzschild-commensurate-v1-collected/result';runs=[base/'schwarzschild-n24-fixed-cube',base/'schwarzschild-n48-fixed-cube',Path(r['lineage'][0]['run'])];body='';limit=r['closed_prefix_time']
 for directory,n,color in zip(runs,[24,48,96],['bad','blue','good']):
  w=waveform(directory);mask=w['time']<=limit;body+=plot(w['time'][mask],w['imag22'][mask],color+',thick',f'$N{n}$');INPUTS[str((directory/'waveform.csv').relative_to(ROOT))]=sha(directory/'waveform.csv')
 for line in r['lineage']:
  directory=Path(line['run']);INPUTS[str((directory/'waveform.csv').relative_to(ROOT))]=sha(directory/'waveform.csv');receipt=directory.parent.parent/'collection-receipt.json';INPUTS[str(receipt.relative_to(ROOT))]=sha(receipt)
 source=E/'radial-oracle/smooth/waveform-upwind-n4800.csv';INPUTS[str(source.relative_to(ROOT))]=sha(source);w=np.genfromtxt(source,delimiter=',',names=True);mask=w['time']<=limit;ct=-1+(np.arange(24)+.5)*2/24;factor=-.5*np.sqrt(15/(32*np.pi))*np.pi*np.sum((1-ct*ct)**2)*2/24;body+=plot(w['time'][mask],factor*w['radial_psi'][mask],'black,dashed,thick','Independent 4800')
 left=axis(body,r'width=8cm,height=6.3cm,xlabel={$t/M$},ylabel={$\Im z_{22}$},xmin=0,xmax='+str(limit)+r',title={Collected waveform, no late extrapolation},legend style={at={(.5,-.32)},anchor=north,legend columns=2}')
 body=''
 for k,color,label in [(0,'blue','$N24/N48$'),(1,'good','$N48/N96$')]:body+=plot([2,6,12],[min(c['orders'][k] for c in r['checks'] if c['time']==t) for t in [2,6,12]],color+',thick,mark=*',label)
 body+=plot([2,12],[1.5,1.5],'black,dashed','Required $1.5$')
 right=axis(body,r'at={(8.5cm,0)},anchor=south west,width=8cm,height=6.3cm,xlabel={$t/M$},ylabel={Minimum pair order, all quantities},xtick={2,6,12},ymin=1.3,ymax=5.3,title={Early field snapshots only},legend pos=south east')
 return pair(left,right)
def main():
 p=argparse.ArgumentParser();p.add_argument('--replace-generated',action='store_true');a=p.parse_args();directory=Path(__file__).resolve().parent;template=directory/'scalar-status-notes.in.tex';target=directory/'test-scalar-status-20261017.tex';manifest=directory/'test-scalar-status-20261017.manifest.json'
 if target.exists():
  if not a.replace_generated or not manifest.exists() or json.loads(manifest.read_text())['outputs'].get(str(target.relative_to(ROOT)))!=sha(target):raise RuntimeError('Existing editable source changed; refusing to overwrite it')
 m=read('cpu-v22/stable-full-review.json');admission=read('minkowski-stable-sbp-cpu-admission.json');gpu=read('aurora/minkowski-stable-gpu-v5-review.json');s=read('cpu-schwarzschild-commensurate-finer/primary-resolved-reference-review.json');r=read('cpu-schwarzschild-dyadic/closed-prefix-t65-review.json');q=read('radial-oracle/smooth/qnm-n4800-review.json')
 continuation=read('aurora/cpu-schwarzschild-dyadic-n96-resume5/submission.json');assert continuation['job_id'].startswith('8939561.')
 for name in ['criteria.json','cpu-schwarzschild-commensurate/primary-review.json','cpu-schwarzschild-commensurate-finer/actual-resolved-rhs-review.json','cpu-schwarzschild-dyadic/reference-selection.json','cpu-schwarzschild-kappa2-diagnostic-review.json','access-recovery-20261017/status.json','access-recovery-20261017/handoff.json','cpu-v22/actual-time-derivative-review.json','cpu-memory-rhs-audit-review.json','spin-zero-qnm-both-branches.json']:read(name)
 assert admission['passed'] and m['passed'] and gpu['passed'] and s['complete'] and not s['field_checks_passed'] and s['finest_accuracy_passed'] and r['closed_prefix_time']==65 and not r['admission'] and q['passed']
 hashes={name:sha(ROOT/'src/test_scalar'/name) for name in ['CMakeLists.txt','main.cpp','geometry.hpp','sbp63.hpp']};assert hashes==admission['delivery_source_sha256']
 for name,digest in hashes.items():INPUTS['src/test_scalar/'+name]=digest
 baseline=read('frozen-baseline-sha256.json');bad=[name for name,digest in baseline.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=digest];assert not bad
 values={'MAPFIG':mapfig(),'DOMAINFIG':domainfig(),'MINKFIG':minkfig(m),'SCHFAILFIG':schfailfig(s),'QNMFIG':qnmfig(s,q),'PARTIALFIG':partialfig(r),'MINKMIN':f"{min(min(c['orders']) for c in m['checks']):.5f}",'TIMEFRAC':texnum(max(c['fraction'] for c in m['timestep'])),'TIMERATIO':f"{max(c['quarter_ratio'] for c in m['timestep']):.5f}",'GPUDELTA':texnum(max(c['scaled_max_field_derivative_difference'] for c in gpu['comparisons'])),'HASHROWS':'\n'.join(r'\texttt{'+name.replace('_',r'\_')+r'} & \texttt{'+digest[:20]+r'} \\' for name,digest in hashes.items())}
 values.update(PARTIALTIME=f"{r['closed_prefix_time']:g}",PARTIALWAVE=texnum(r['partial_independent_waveform_relative_L2']))
 text=template.read_text()
 for key,value in values.items():text=text.replace('@@'+key+'@@',value)
 if '@@' in text:raise RuntimeError('Unresolved template placeholder')
 header='% Generated from preserved evidence; no simulation or admission added.\n% See adjacent manifest for full input hashes.\n'
 target.write_text(header+text)
 record=dict(created=time.time(),scope='Dated research notes only; no new evolution, submission or scientific admission. All figures embedded as vector TikZ/PGFPlots in standalone source.',scientific_admission_added=False,native_trajectories_added=0,protected_baseline_files=len(baseline),protected_baseline_mismatches=bad,input_sha256=INPUTS,template_sha256=sha(template),generator_sha256=sha(Path(__file__)),outputs={str(target.relative_to(ROOT)):sha(target)},literature_primary_urls=['https://arxiv.org/abs/'+v for v in ['0712.4333','0906.3342','1104.4199','1409.1887','1910.13452','1908.10377']])
 manifest.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(source=str(target),manifest=str(manifest),protected_baseline_files=len(baseline),inputs=len(INPUTS),figures=6),indent=2))
if __name__=='__main__':main()
