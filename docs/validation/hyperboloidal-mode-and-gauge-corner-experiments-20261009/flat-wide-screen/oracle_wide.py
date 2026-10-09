"""Independent 100-digit wide-reference oracle, with factored Omega prime."""
from pathlib import Path
import hashlib,json,math,subprocess,time
import mpmath as mp
mp.mp.dps=100;w=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r0,r1=map(mp.mpf,[.05,.95]);S=mp.mpf(1);a=mp.mpf(.5)
def omega(r):
 if r<=r0:return mp.mpf(1)
 out=(S-r)*(S+r)/(2*a*S)
 if r>=r1:return out
 s=(r-r0)/(r1-r0);t=(r1-r)/(r1-r0);g=-1/s+1/t;e=mp.exp(-abs(g));v=e/(1+e) if g<=0 else 1/(1+e)
 return 1-v+v*out
def op(r):
 if r<=r0:return mp.mpf(0)
 if r>=r1:return -r/(a*S)
 width=r1-r0;s=(r-r0)/width;t=(r1-r)/width;g=-1/s+1/t;e=mp.exp(-abs(g));v=e/(1+e) if g<=0 else 1/(1+e)
 # Factoring v avoids differentiating an Omega value rounded identically1.
 gp=(1/s**2+1/t**2)/width;out=(S-r)*(S+r)/(2*a*S)
 return -v*((1-v)*gp*(1-out)+r/(a*S))
def ell(r):return omega(r)-r*op(r)
def b(r):return mp.sqrt((-r*op(r))*(2*omega(r)-r*op(r)))
def kp(r):return -(omega(r)*(mp.diff(b,r)+2*b(r)/r)-3*b(r)*op(r))/ell(r)
def ar(r):return mp.mpf(2)/3*(-mp.diff(b,r)+b(r)/r)/ell(r)
radii=[0.,.05,math.nextafter(.05,.95),.050001,.0501,.051,.06,.1,.3,.5,.7,.9,.949,.949999999999,math.nextafter(.95,.05),.95,.99]
radii += [.05+.9*j/100 for j in range(1,100)]
for ng in [700.,745.,750.,1000.,1400.,1490.,1500.,1550.,1600.]:radii.append(.05+.9*2/(ng+2+math.hypot(ng,2)))
text=''.join(format(x,'.17g')+'\n' for x in radii);(w/'radial-input.txt').write_text(text)
started=time.monotonic();p=subprocess.run([str(w/'export-radial')],input=text,capture_output=True,text=True);assert p.returncode==0,p.stderr
(w/'radial-values.jsonl').write_text(p.stdout);(w/'radial-export.stderr').write_text(p.stderr)
names=['Omega','Omega_prime','Omega_second','alpha','alpha_prime','alpha_second','b','b_prime','b_second','Kphysical','Kphysical_prime','Ar','Ar_prime','Kbar','outgoing','ingoing']
errors=[];tails=[]
for line in p.stdout.splitlines():
 row=json.loads(line);r=mp.mpf(row['r']);v=row['values'];assert all(math.isfinite(x) for x in v)
 if r<=r0 or r>=r1:continue # Core/outer byte-parity is checked by actual C++ gate.
 O=omega(r);Op=op(r);Opp=mp.diff(op,r);O3=mp.diff(op,r,2);L=ell(r);B=b(r);Bp=mp.diff(b,r);Bpp=mp.diff(b,r,2)
 exact=[O,Op,Opp,L,-r*Opp,-Opp-r*O3,B,Bp,Bpp,kp(r),mp.diff(kp,r),ar(r),mp.diff(ar,r),-(Bp+2*B/r)/L,L+B,-O*O/(L+B)]
 for name,actual,want in zip(names,v,exact):
  rounded=float(want);delta=abs(actual-rounded);relative=delta/abs(rounded) if rounded else 0.
  # The inner boost must retain relative accuracy where resolved, including
  # representable derivatives after the cutoff/boost value underflows.
  strict=name in ['b','b_prime','b_second'] and r<(r0+r1)/2
  allowance=max(4*float.fromhex('0x0.0000000000001p-1022'),abs(rounded)*5e-9) if strict else 3e-9*max(1.,abs(rounded))
  assert delta<=allowance,(row['r'],name,actual,mp.nstr(want,30),delta,allowance)
  errors.append({'r':row['r'],'field':name,'absolute_error':delta,'normalized_error':delta/(1+abs(rounded)),'relative_error':relative,'strict_inner_boost':strict})
 if v[6]==0 and (v[7]!=0 or v[8]!=0):tails.append(row)
assert tails,'No representable derivative-after-boost-underflow tail was tested'
result={'scope':'Independent wide(.05,.95) exponent-one supplement to frozen flat-height family, not a new geometry/stability claim',
 'mpmath_dps':mp.mp.dps,'checked_transition_points':len(errors)//len(names),'checked_scalar_values':len(errors),
 'max_normalized_error':max(x['normalized_error'] for x in errors),
 'max_absolute_error':max(errors,key=lambda x:x['absolute_error']),
 'max_resolved_inner_boost_relative_error':max((x for x in errors if x['strict_inner_boost'] and x['absolute_error']>0 and x['relative_error']<1),key=lambda x:x['relative_error']),
 'derivative_survives_boost_underflow':tails,'errors':errors,'seconds':time.monotonic()-started,
 'long_double_mantissa_bits':53,'source_sha256':sha(Path(__file__)),'export_executable_sha256':sha(w/'export-radial')}
(w/'oracle-wide.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='errors'},indent=2))
