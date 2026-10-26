#!/usr/bin/env python3
"""Probe the actual cell-aligned trumpet excision faces; do not evolve fields."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from audit_geometry import conformal

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise RuntimeError('Existing characteristic audit is immutable')
 records=[]
 for spin in (0,.5,.9):
  for n,cut in ((24,.126),(32,.126),(40,.126),(40,.20),(32,.06),(40,.06)):
   dx=3/n;removed=sum((j+.5)*dx<cut for j in range(n));b=removed*dx;minimum=1e99;location=None
   if not removed:records.append(dict(spin=spin,n=n,requested_cut=cut,actual_half_width=b,has_cut=False));continue
   for A in range(3):
    tangent=[j for j in range(3) if j!=A]
    for sign in (-1,1):
     normal=np.zeros(3);normal[A]=-sign
     for y in np.linspace(-b,b,21):
      for z in np.linspace(-b,b,21):
       x=np.zeros(3);x[A]=sign*b;x[tangent]=[y,z];G=np.linalg.inv(conformal(x,1,spin));bn=G[0,1:]@normal/G[0,0];disc=bn*bn-normal@G[1:,1:]@normal/G[0,0];minus=bn-np.sqrt(disc)
       if minus<minimum:minimum=float(minus);location=x.tolist()
   records.append(dict(spin=spin,n=n,requested_cut=cut,actual_half_width=b,has_cut=True,min_cminus=minimum,worst_location=location,all_outflow=minimum>=-1e-12))
 result=dict(scope='Independent value-only physical trumpet pullback; dense 21x21 probes on each actual removed-cube face, including all edges and corners. No scalar time evolution.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),provider_sha256=hashlib.sha256(Path(__file__).with_name('audit_geometry.py').read_bytes()).hexdigest(),records=records)
 a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
