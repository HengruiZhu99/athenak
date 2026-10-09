from pathlib import Path
import hashlib,json,time,subprocess
import sympy as s
from sympy.polys.matrices import DomainMatrix
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
F=ROOT/'build-layer-research/continuum/q-null-curvature-ideal/immutable-Q-Einstein-cubic-null-curvature-timejet-20261009'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(F/'index.json')=='0302a9da72f1a36f5fd6bf7dc9311fccc08c97ac1eace3282423a2ee38517f13'
for f in json.loads((F/'index.json').read_text())['files']:assert sha(F/f['path'])==f['sha256']
inputs={str(f.relative_to(ROOT)):sha(f)for f in [F/'index.json',F/'actual-release.json',P/'check_tensor.py']}
start=time.monotonic();j=json.loads((F/'actual-release.json').read_text());rat=lambda x:s.Rational(str(x)).limit_denominator(1000000)
out={'rows':[]}
for data in j['radii']:
    a=rat(data['a']);cols=data['orientations'][0]['columns']
    E=s.Matrix([c['E']for c in cols]).T.applyfunc(rat);R=s.Matrix([[rat(c['delta_Rq'])for c in cols]])
    B=E.col_join(R);rr,piv=DomainMatrix.from_Matrix(B).convert_to(s.QQ).rref();rr=rr.to_Matrix();assert len(piv)==128
    # q_AB(Omega)=r^2*bargamma_AB in fixed angular labels. Its first-normal TF
    # coefficient is (bargamma_AB,1-2a*bargamma_AB,0)^TF. Components here are
    # fixed local Cartesian axes at the base point, with X the outward offset.
    h=s.zeros(2,400);h[0,7]=s.Rational(1,2);h[0,10]=1;h[1,11]=1
    q1=-2*a*h;q1[0,27]=-a/2;q1[0,30]=-a;q1[1,31]=-a
    A=s.zeros(2,400);A[0,12]=s.Rational(1,2);A[0,15]=1;A[1,16]=1
    S=s.zeros(2,400);S[0,48]=s.Rational(1,2);S[0,69]=-s.Rational(1,2);S[1,49]=s.Rational(1,2);S[1,68]=s.Rational(1,2)
    ix=lambda k:j['labels'].index(f'R0_{k}[0,0,0]')
    shear=s.Matrix.vstack((E[ix(12),:]+2*E[ix(15),:])/2,E[ix(16),:])
    assert A-q1/(2*a)-2*h-S == -a*a*shear/2
    free=[k for k in range(400)if k not in piv];K=s.zeros(400,len(free))
    for col,k in enumerate(free):
        K[k,col]=1
        for i,p in enumerate(piv):K[p,col]=-rr[i,k]
    assert B*K==s.zeros(B.rows,K.cols)
    image=q1*K;_,columns=DomainMatrix.from_Matrix(image).convert_to(s.QQ).rref();assert len(columns)==2
    V=K[:,list(columns)]*image[:,list(columns)].inv()
    assert q1*V==s.eye(2) and B*V==s.zeros(B.rows,2)
    assert A*V-q1*V/(2*a)-2*h*V-S*V==s.zeros(2,2)
    out['rows'].append({'a':float(a),'compatible_frame_rank':128,'compatible_frame_nullity':272,
        'two_TF_first_normal_cut_metric_image_rank':2,'surjective':True,
        'shear_identity':'A_TF-q1_TF/(2a)-2h0_TF-S=-a^2 R0_A_TF/2; S=.5(sym tangential derivative of fixed-frame h_nA)^TF',
        'preimages':[{'target':[int(k==i)for k in range(2)],'nonzero_coefficients':[[k,str(x)]for k,x in enumerate(V[:,i])if x],
            'A_boundary_TF':[str(x)for x in(A*V)[:,i]],'h0_TF':[str(x)for x in(h*V)[:,i]],'normal_frame_gradient_TF':[str(x)for x in(S*V)[:,i]]}for i in range(2)]})
out.update({'passed_finite_jet_tensor_freedom_probe':True,'scope':'Surjectivity onto two cut-metric first-normal TF coefficients in the frozen necessary finite linear Einstein/null/shear+roundness kernel; corresponding A is displayed through the imposed shear identity. No genuine radiative Weyl data, exact Einstein germ, hierarchy closure, native/global evolution or source adoption.',
    'inputs_before':inputs,'inputs_after':{f:sha(ROOT/f)for f in inputs},'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'seconds':time.monotonic()-start})
assert out['inputs_before']==out['inputs_after'];(P/'receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
