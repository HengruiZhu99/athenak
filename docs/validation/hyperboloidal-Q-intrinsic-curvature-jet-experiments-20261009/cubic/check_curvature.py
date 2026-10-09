from pathlib import Path
import hashlib,json
import sympy as s
from sympy.polys.matrices import DomainMatrix

P=Path(__file__).resolve().parent
j=json.loads((P/'actual-release.json').read_text())
rat=lambda x:s.Rational(str(x)).limit_denominator(1000000)
exponents=[(i,k,d-i-k)for d in range(4)for i in range(d,-1,-1)for k in range(d-i,-1,-1)]
labels=j['labels'];out={'rows':[]};reconstruction_error=0.;rotation_error=0.;curvature_error=0.

def reconstructed(values):
    global reconstruction_error
    M=s.Matrix(values);R=M.applyfunc(rat)
    reconstruction_error=max(reconstruction_error,max(abs(float(R[i,k])-float(M[i,k]))for i in range(M.rows)for k in range(M.cols)))
    return R

def reduction(E):
    rr,piv=DomainMatrix.from_Matrix(E).convert_to(s.QQ).rref();rr=rr.to_Matrix()
    def remainder(M):
        M=M.copy()
        for i,p in enumerate(piv):M-=M[:,p]*rr[i,:]
        return M
    return remainder,rr,piv

# Independently derived induced-metric variation formula in the sphere's
# graph coordinates: deltaR=sum_AB partial_A partial_B h_AB
# -Delta(sum_A h_AA)-2sum_A h_AA, where h_AB is the induced perturbation.
# The explicit Cartesian-to-induced second derivative below retains normal
# frame derivatives; no actual 2D curvature implementation is reused.
def curvature_formula(monomial,field):
    power=exponents[monomial];T=s.zeros(3)
    if field==1:T=-s.eye(3)
    if 7<=field<12:
        i,k=[(0,0),(0,1),(0,2),(1,1),(1,2)][field-7]
        T[i,k]=1;T[k,i]=1
        if i==k:T[2,2]=-1
    def derivative(indices):
        p=list(power);v=1
        for k in indices:
            if p[k]==0:return 0
            v*=p[k];p[k]-=1
        return v if not any(p)else 0
    def h(i,k,*d):return T[i,k]*derivative(d)
    def induced_second(A,B,C,D):
        return (h(A,B,C,D)-(C==D)*h(A,B,0)
            -(A==C)*h(0,B,D)-(A==D)*h(0,B,C)
            -(B==C)*h(A,0,D)-(B==D)*h(A,0,C)
            +((A==C and B==D)+(A==D and B==C))*h(0,0))
    return sum(induced_second(A,B,A,B)-induced_second(A,A,B,B)for A in [1,2]for B in [1,2])-2*(h(1,1)+h(2,2))

for radius in j['radii']:
    a=rat(radius['a']);columns=radius['orientations'][0]['columns']
    other=radius['orientations'][1]['columns']
    for c,d in zip(columns,other):
        assert(c['monomial'],c['field'])==(d['monomial'],d['field'])
        for key in ['E','next_R0']:
            rotation_error=max(rotation_error,max(abs(x-y)for x,y in zip(c[key],d[key])))
        for key in ['N1_t','N0_t','Q0_t','delta_Rq','delta_Rq_t','N1_t_regular_piece','N1_t_pole_piece']:
            rotation_error=max(rotation_error,abs(c[key]-d[key]))
        curvature_error=max(curvature_error,abs(c['delta_Rq']-float(curvature_formula(c['monomial'],c['field']))))
        if sum(exponents[c['monomial']])==3:
            assert c['N1_t']==c['N1_t_regular_piece']==c['N1_t_pole_piece']==0
    E=reconstructed([c['E']for c in columns]).T
    L=reconstructed([[c['N1_t']for c in columns]])
    R=reconstructed([[c['delta_Rq']for c in columns]])
    rates=reconstructed([[c[key]for c in columns]for key in ['N0_t','Q0_t','delta_Rq_t']])
    B=reconstructed([c['next_R0']for c in columns]).T
    rem,rr,piv=reduction(E)
    assert len(piv)==127
    assert rem(L+R/a**2)==s.zeros(1,400)
    assert rem(rates)==s.zeros(3,400) and rem(B)==s.zeros(20,400)
    assert rem(L)!=s.zeros(1,400) and rem(R)!=s.zeros(1,400)
    # A smaller set confirms the null/curvature identity is not manufactured
    # by imposing unavailable higher derivatives or next-R0 as initial data.
    small_rows=[]
    for i,label in enumerate(labels):
        name=label.split('[')[0];degree=sum(map(int,label.split('[')[1].strip(']').split(',')))
        if ((name=='H'and degree<=2)or(name.startswith(('M','Z'))and degree<=1)
            or(name=='Theta'and degree<=2)or(name in ['N','Qnum']and degree<=2)
            or(name.startswith('R0_')and degree<=1)):
            small_rows.append(i)
    small=E[small_rows,:];rsmall,_,psmall=reduction(small)
    assert len(psmall)==59 and rsmall(L+R/a**2)==s.zeros(1,400)
    rp=DomainMatrix.from_Matrix(small.T).convert_to(s.QQ).rref()[1]
    basis=small[list(rp),:]
    coefficients=(L+R/a**2)[:,list(psmall)]*basis[:,list(psmall)].inv()
    identity=[{'condition':labels[small_rows[rp[i]]],'coefficient':str(coefficients[i])}for i in range(len(rp))if coefficients[i]]
    assert all(not row['condition'].startswith(('Qnum','R0_'))for row in identity)
    # Exact finite compatible vectors demonstrate that intrinsic roundness
    # does not impose every A/Lambda/metric boundary value zero.
    full=E.col_join(R);rf,rrf,pf=reduction(full);assert len(pf)==128
    controls=[]
    for family,target_indices in [('finite_A',list(range(12,17))),('finite_Lambda',list(range(17,20))),('finite_metric',list(range(7,12)))]:
        chosen=None
        for free in range(400):
            if free in pf:continue
            v=s.zeros(400,1);v[free]=1
            for i,pivot in enumerate(pf):v[pivot]=-rrf[i,free]
            targets=[k for k in target_indices if v[k]]
            if targets:
                v/=v[targets[0]];chosen=v;break
        assert chosen is not None
        assert E*chosen==s.zeros(E.rows,1)and R*chosen==s.zeros(1,1)
        assert L*chosen==s.zeros(1,1)and B*chosen==s.zeros(20,1)
        controls.append({'family':family,'normalization_index':targets[0],
            'nonzero_coefficients':[[i,str(x)]for i,x in enumerate(chosen)if x],
            'scope':'Necessary finite Einstein/null/shear jets; no exact vacuum germ or radiative interpretation asserted.'})
    out['rows'].append({'a':float(a),'columns':400,'condition_rows':246,'E_rank':len(piv),'E_nullity':400-len(piv),
        'E_plus_intrinsic_curvature_rank':len(pf),'curvature_restriction_nullity':400-len(pf),
        'N1_plus_Rq_over_a2_in_E_rowspace':True,'N0_Q0_Rq_time_and_next20R0_in_E_rowspace':True,
        'pure_cubic_N1_regular_and_pole_columns_exact_zero':True,'lower_condition_rows':len(small_rows),'lower_condition_rank':len(psmall),
        'explicit_lower_order_identity':identity,'finite_boundary_value_controls':controls})

oracle=j['oracles'];assert oracle['series_algebra_and_chart_identity_max']<1e-12
pts=oracle['point_checks'];assert len(pts)==3
for key,order in [('regular_and_pole_Taylor_error',16),('physical_constraints_Taylor_error',16),('assembled_Ndot_Taylor_error',8)]:
    for i in range(2):assert order*.9<pts[i][key]/pts[i+1][key]<order*1.1
fds=oracle['dual_finite_difference'];assert fds[-1]['regular_and_pole_error']<1e-8 and fds[-1]['assembled_Ndot_error']<2e-6
assert fds[-1]['regular_and_pole_error']<fds[0]['regular_and_pole_error']/1000
assert fds[-1]['assembled_Ndot_error']<fds[0]['assembled_Ndot_error']/1000
assert rotation_error<1e-10 and reconstruction_error<1e-10 and curvature_error<1e-12
out.update({'passed_local_finite_jet_gate':True,'reference_audited_radii':[r['a']for r in out['rows']],
    'actual_basis_columns':3200,'condition_rows_per_radius':246,
    'rational_reconstruction_max_error':reconstruction_error,'independent_orientation_max_error':rotation_error,
    'independent_intrinsic_curvature_formula_max_error':curvature_error,'oracles':oracle,
    'scope':'Exact rowspace statements of rationally reconstructed analytic-reference matrices, not exact nonlinear assembly. Conditional joint first-time null/curvature tangency on stated finite Einstein/null/shear jets; full compatibility ideal evolution, radiative retention and finite-Omega/global stability unproved.',
    'new_gauge_or_native_admission':False,'all_metric_A_Lambda_Omega_falloffs_imposed':False})
(P/'check-report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
