#!/usr/bin/env python3
"""Oblique shifted-wave reflection model, not a native boundary experiment."""
import argparse,hashlib,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('lift_review',type=Path);p.add_argument('output',type=Path);a=p.parse_args();j=json.loads(a.lift_review.read_text());records=[]
def wave(beta,c,degrees):
 theta=math.radians(degrees);ko=math.cos(theta);kt=math.sin(theta);omega=-beta*ko+c;aa=c*c-beta*beta;kr=2*beta*omega/aa-ko
 dispersion=lambda k:abs((omega+beta*k)**2-c*c*(k*k+kt*kt))/max(1,(omega+beta*k)**2,c*c*(k*k+kt*kt))
 group=lambda k:-beta+c*k/math.sqrt(k*k+kt*kt)
 bo=-omega+(c-beta)*ko;br=-omega+(c-beta)*kr;R=-bo/br;fo=omega*(aa*ko-beta*omega);fr=omega*(aa*kr-beta*omega);ratio=R*R*abs(fr/fo)
 passed=max(dispersion(ko),dispersion(kr))<=1e-10 and group(ko)>0 and group(kr)<0 and abs(ratio-R*R)<=1e-10*max(1,R*R) and (degrees!=0 or abs(R)<=1e-12)
 return dict(angle_degrees=degrees,amplitude_reflection=abs(R),energy_flux_reflection=ratio,normal_wavenumber_stretch=abs(kr/ko),normal_derivative_amplitude_ratio=abs(kr*R/ko),outgoing_normal_group_speed=group(ko),reflected_normal_group_speed=group(kr),maximum_dispersion_error=max(dispersion(ko),dispersion(kr)),passed=passed)
for x in j['records']:
 beta=-(x['lo']+x['hi'])/2;c=(x['hi']-x['lo'])/2;tests=[wave(beta,c,t) for t in [0,15,30,45,60,75]];records.append(dict(source=x['source'],sample=x['sample'],omega_conformal=x['omega'],beta=beta,c=c,tests=tests,passed=all(t['passed'] for t in tests)))
aux=[wave(0.,1.,t) for t in [0,15,30,45,60,75]];aux_pass=all(abs(x['amplitude_reflection']-math.tan(math.radians(x['angle_degrees'])/2)**2)<=1e-12 for x in aux)
r=dict(passed=all(x['passed'] for x in records) and aux_pass,records=records,zero_shift_analytic_control_passed=aux_pass,zero_shift_controls=aux,scope='Frozen scalar subsidiary-wave principal reflection only. Uses native normal speed pairs with isotropically rescaled tangent coordinates; does not measure actual angular native boundary reflection or prove cause of collar errors.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=hashlib.sha256(a.lift_review.read_bytes()).hexdigest());a.output.write_text(json.dumps(r,indent=2)+'\n');print('passed',r['passed']);print('45degrees',[(x['sample'],x['tests'][3]['amplitude_reflection'],x['tests'][3]['normal_wavenumber_stretch'],x['tests'][3]['normal_derivative_amplitude_ratio']) for x in records])
