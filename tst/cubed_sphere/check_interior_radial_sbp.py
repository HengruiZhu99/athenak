#!/usr/bin/env python3
"""Exact obstruction to local-boundary diagonal-norm SBP for candidate D1.
No statement about dense norms, nonlocal traces, or nonlinear PDE stability.
"""
import re,json
from fractions import Fraction as F
from pathlib import Path
text=Path('src/mesh/cubed_sphere/interior_radial.hpp').read_text().split('InteriorRadialSecond')[0]
coeff=[F(int(a),int(b)) for a,b in re.findall(r'Real\((-?\d+)\)/Real\((\d+)\)',text)]
assert len(coeff)==18
# i=last, j=last-5. Row j uses unchanged radius-two centered derivative.
Dii=coeff[5];Dij=coeff[0];Dji=F(0);Djj=F(0)
assert Dii>0 and Dij!=0 and Dji==Djj==0
out=dict(scope='Diagonal positive H; HD+D^T H=tR tR^T-tL tL^T; local disjoint endpoint traces, left trace zero on last six nodes',Dii=str(Dii),Dij=str(Dij),Dji=str(Dji),Djj=str(Djj),diagonal_norm_sbp_possible=False,proof=['At j,j: 0=tR[j]^2, so tR[j]=0.','At i,j: H[i]*Dij + H[j]*0 = tR[i]*0 = 0.','Since H[i]>0 and Dij=-1/5, this is impossible.'],limitations='Does not exclude dense-norm constructions or redesigned stencils. Does not prove instability of every bounded finite experiment.')
path=Path('docs/validation/cubed-sphere-balanced-20261013/outer-closure/sbp-obstruction.json');path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
