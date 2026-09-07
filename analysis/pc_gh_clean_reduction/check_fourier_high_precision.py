#!/usr/bin/env python3
"""Independent 60-digit exponential at the largest sampled transient."""
import argparse
import json
from pathlib import Path
import sys
import mpmath as mp
import numpy as np
import scipy.linalg as la
import sympy as s
sys.dont_write_bytecode=True
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-dir',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();sys.path.insert(0,str(a.reference_dir))
from exact_flat_symbol import matrices
from check_flat_fourier import source_matrix
cases=json.loads(a.cases.read_text());_,params,k,t,expected=max((point['norm2'],tuple(c['parameters']),row['k'],point['t'],point['norm2']) for c in cases for row in c['rows'] for point in row['times'])
mp.mp.dps=60;rate,eta,kap=params
P=matrices(s.Integer(rate),s.Integer(eta));J=source_matrix(s.Integer(rate),s.Integer(eta),s.Integer(kap));A=J+s.I*s.Rational(str(k))*sum(((i+1)*P[i] for i in range(3)),s.zeros(50))/s.sqrt(14)
Am=mp.matrix([[mp.mpc(str(s.re(A[i,j]).evalf(65)),str(s.im(A[i,j]).evalf(65))) for j in range(50)] for i in range(50)])
X=mp.expm(mp.mpf(str(t))*Am);norm=float(la.svdvals(np.array(X.tolist(),dtype=complex))[0]);error=abs(norm-expected)/(1+abs(expected));assert error<=2e-9
out=dict(status='PASS',decimal_digits=60,parameters=params,k=k,t=t,high_precision_exponential_norm=norm,reported_scipy_norm=expected,normalized_norm_difference=error,scope='60-digit full matrix exponential with double SVD norm; largest sampled point only')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
