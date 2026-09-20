from pathlib import Path
import re,json,hashlib
root=Path(__file__).resolve().parent
repo=root.parents[2]
base=(repo/'review/stability-isolation-20260919/gauge-slow/shift_Gamma2/input.athinput').read_text()
def setv(s,block,key,value):
 pat=rf'(<{re.escape(block)}>\n)(.*?)(?=\n<|\Z)'
 m=re.search(pat,s,re.S); assert m,block
 body=m[2]
 line=rf'(?m)^{re.escape(key)}\s*=.*$'
 if re.search(line,body):body=re.sub(line,f'{key} = {value}',body)
 else:body+=f'\n{key} = {value}\n'
 return s[:m.start()]+m[1]+body+s[m.end():]
cases={
 'small_dx025':dict(L=2,n=16,nb=8,dx=.25),
 'wide_dx025':dict(L=4,n=32,nb=16,dx=.25),
 'small_dx0125':dict(L=2,n=32,nb=16,dx=.125),
}
for name,c in cases.items():
 s=base
 for a in (1,2,3):
  for k,v in {f'nx{a}':c['n'],f'x{a}min':-c['L'],f'x{a}max':c['L']}.items():s=setv(s,'mesh',k,v)
  s=setv(s,'meshblock',f'nx{a}',c['nb'])
 for b,k,v in [('time','cfl_number',.15),('time','ndiag',200),('time','tlim',1000),('output2','dt',25),('output3','dt',25),('output4','dt',100),('z4c','characteristic_bc_diagnostics','false')]:s=setv(s,b,k,v)
 p=root/(name+'.athinput');p.write_text(s)
 c['input_sha256']=hashlib.sha256(s.encode()).hexdigest()
 c['dt_expected']=.15*c['dx']
(root/'cases.json').write_text(json.dumps({'cases':cases,'gauge':{'shift_Gamma':2,'residual_lapse_f':1},'description':'Matched vacuum pulse domains/resolution with conservative CFL.15, sixth-order, original boundaries, no sponge, no matter feedback. Target1000M; walltime stopping is incomplete. Both exact zero and perturbation stability required before spin extension.'},indent=2)+'\n')
print(root)
