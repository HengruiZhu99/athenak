#!/usr/bin/env python3
"""Prepare short A-only single-hole controls; no evolution sources are changed."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('output',type=Path)
p.add_argument('--base-cycles',type=int,default=510)
p.add_argument('--duration-reason',default='Frozen-background outgoing travel .05419-.08240 M; compare through .718246 M after the observed early pulse.')
args=p.parse_args();assert args.base_cycles>0 and args.base_cycles%3==0
args.output.mkdir(exist_ok=False)
base=(ROOT/'tst/z4c_hyperboloidal/aurora/n12.athinput').read_text()
base=base[:base.index('<fastflow>')]
for n in (12,16,20):
 for family in ('control','gauge'):
  for half in (False,True) if n==12 and family=='gauge' else (False,):
   name=f'{family}-N{n}'+('-halfdt' if half else '')
   dt=0.00046944203735617445*12/n/(2 if half else 1)
   # Integer-scaled cycle targets provide matching times at all N and dt.
   cycles=args.base_cycles*n//12*(2 if half else 1)
   text=base.replace('native_n12_5M',name)
   for axis in (1,2,3):
    text=text.replace(f'nx{axis} = 36',f'nx{axis} = {3*n}').replace(f'nx{axis} = 12',f'nx{axis} = {n}')
   text=text.replace('nlim = 3551',f'nlim = {cycles}').replace('dt = 0.00046944203735617445',f'dt = {dt:.17g}').replace('ndiag = 1','ndiag = 100')
   text+='\nscri_treatment=A\noverlap_mode=agreement_full\n'
   text+='\n<output1>\nfile_type=hst\ndt=.01\ndata_format=%.17e\nhyperboloidal_scri=true\nhyperboloidal_scri_investigation=true\n'
   text+='\n<output2>\nfile_type=rst\ndt=10\nsingle_file_per_rank=false\n'
   text+=f'\n<output3>\nfile_type=scri\ndt=-1\ndcycle={2 if half else 1}\nmodes=false\nboundary_diagnostics=true\nunit_description=G=c=1; M0=1/3; single Schwarzschild hole\n'
   (args.output/(name+'.athinput')).write_text(text)
(args.output/'cases.json').write_text(json.dumps({'mass':1/3,'amplitude':.001,'center':.9555,'half_width':.009,'base_cycles':args.base_cycles,'target_M':3*args.base_cycles*.00046944203735617445,'reason':args.duration_reason},indent=2))
