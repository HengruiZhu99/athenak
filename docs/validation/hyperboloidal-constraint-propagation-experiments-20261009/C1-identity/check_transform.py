"""Exact storage transformation and independent actual full-tensor identities."""
from pathlib import Path
import json
import sympy as s

p = Path(__file__).resolve().parent
O,alpha,chi,theta,P,w,Aij,gamma,kap,k2 = s.symbols('Omega alpha chi Theta P w Aij gamma kappa_input kappa2')
za,zc,zo = s.symbols('Ztilde_dalpha Ztilde_dchi Ztilde_dOmega')
Kbar = (P-3*w)/O
Tbar = theta/O
paper_delta_K = 2*chi*za-2*alpha*chi*zo/O
paper_delta_Theta = -alpha*Tbar*(Kbar+2*Tbar)-chi*za-alpha*zc/2
delta_P = s.factor(O*paper_delta_K)
delta_theta = s.factor(O*paper_delta_Theta)
assert s.simplify(delta_P-2*chi*(O*za-alpha*zo)) == 0
assert s.simplify(delta_theta+alpha*theta*(P+2*theta-3*w)/O+O*chi*za+alpha*O*zc/2) == 0
K = P+2*theta
delta_Kphysical = s.expand(delta_P+2*delta_theta)
physical_Kij = Aij/(O*chi)+gamma*K/3
delta_A = -2*alpha*Aij*theta/O
delta_Kij = delta_A/(O*chi)+gamma*delta_Kphysical/3
B0 = alpha*O*zc+2*alpha*chi*zo-(6*alpha*w+3*(1+k2)*kap)*theta/O
target_delta = -2*alpha*theta*physical_Kij/O-gamma*(1+k2)*kap*theta/O-gamma*B0/3
assert s.simplify(delta_Kij-target_delta) == 0

# Covariant physical divergence and lapse-gradient terms cancel the unwanted
# Omega-gradient contraction in physical Theta propagation.
divt = s.symbols('div_tilde_Ztilde')
physical_divergence = O**2*chi*divt-O*chi*zo-O**2*zc/2
physical_Z_dA = O*chi*za-alpha*chi*zo
covariant_spatial = alpha*physical_divergence/O-physical_Z_dA
assert s.simplify(covariant_spatial-(alpha*O*chi*divt-O*chi*za-alpha*O*zc/2)) == 0
gi_alpha,gi_Omega = s.symbols('gtilde_inverse_dalpha gtilde_inverse_dOmega')
delta_Lambda = -2*theta*gi_alpha/O+2*alpha*theta*gi_Omega/O**2
assert s.simplify(O**2*delta_Lambda+2*O*theta*gi_alpha-2*alpha*theta*gi_Omega) == 0

rows = json.loads((p/'tensor.json').read_text())
keys = [k for k in rows[0] if k.endswith('_error')]
maxima = {k:max(row[k] for row in rows) for k in keys}
assert len(rows) == 384
for k,v in maxima.items():
    assert v < 2e-10, (k,v)
assert max(row['Einstein_sector_addition'] for row in rows) == 0
assert max(row['C0_switch_addition'] for row in rows) == 0
affine = json.loads((p/'affine.json').read_text())
assert affine['AppendixB_C1_addition_norm'] == 0
assert abs(affine['noncovariant_Z_rate']-1) < 1e-14
assert abs(affine['extra_shift_rate']-1) < 1e-14
assert affine['repaired_Z_error'] < 1e-14
principal = json.loads((p/'principal.log').read_text())
assert principal['passed_kernel_cases'] == 360
assert principal['harmonic_endpoint_complete']
report = {'passed_tensor_and_transform':True, 'candidate_native_accepted':False,
    'fulltensor_rows':len(rows), 'relative_or_normalized_errors':maxima,
    'absolute_physical_residual_maxima':{k:max(row[k] for row in rows)
        for k in rows[0] if k.endswith('_absolute_residual')},
    'smallest_positive_Omega':min(row['Omega'] for row in rows),
    'Einstein_sector_additions_exact_zero':True,'C0_switch_exact_zero':True,
    'AppendixB_affine_shift_counterexample':affine,
    'actual_full20_principal_gate':principal,
    'exact_delta_P':str(delta_P),'exact_delta_Theta':str(delta_theta),
    'exact_delta_Kphysical':str(s.factor(delta_Kphysical)),
    'exact_delta_Lambda':str(delta_Lambda),
    'regularity':'Physical Theta is unrestricted. Lambda contains an actual '
                 '2 alpha Theta gtilde_inverse dOmega / Omega^2 addition. '
                 'No Omega floor, Theta falloff or scri assembly is provided.',
    'scope':'Mechanical AppendixB C1 matches physical Kij/Theta Z4 identities '
            'but retains the printed-system extra Z shift-gradient term. '
            'The separately derived connection subtraction restores the '
            'physical covector identity at finite Omega. Principal-only '
            'equality does not establish offconstraint pole or native stability.'}
(p/'check-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
