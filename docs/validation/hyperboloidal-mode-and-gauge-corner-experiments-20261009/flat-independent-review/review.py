"""Read-only symbolic review of the already tested flat-height p1 family."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

import sympy as s

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
F = ROOT / 'build-layer-research/continuum/flat-penrose-height/immutable-flat-penrose-math-v2-20261009'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
start = time.monotonic()
wanted = 'cc8a2983de6a2e91f0ff55f4bad6928c0524fe3109e88060be4959407bf48f5e'
assert sha(F / 'index.json') == wanted
catalog = json.loads((F / 'index.json').read_text())
for name, q in catalog['files'].items():
    assert sha(F / name) == q['sha256']
    assert (F / name).stat().st_size == q['bytes']
assert sha(F / 'prior-family/flat_power.hpp') == '7d3c22bb5e23627a5da83d542fd4100d109fad43612ee3f995c223bd50df1135'

r, O, Op, Opp, b, bp, w, gp, E, D = s.symbols('r O Op Opp b bp w gp E D', nonzero=True)
L = O - r * Op
Lp = -r * Opp
b2 = L**2 - O**2
bbp = L * Lp - O * Op
# Pull back Omega^2(-dT^2+dR^2+R^2 dOmega_sphere^2),
# with R'=L/Omega^2 and h_R=b/L.
metric = s.factor((1 - b**2 / L**2) * L**2 / O**2)
assert s.simplify(s.expand(metric).subs(b**2, b2) - 1) == 0
assert s.simplify((O**2 + b2) - L**2) == 0
# Stationary Penrose spatial metric delta gives Kbar=(Lie_beta delta)/(2L).
kr = (-O * bp + b * Op) / L
kt = (-O * b / r + b * Op) / L
assert s.factor(kt + b / r) == 0
P = kr + 2 * kt
assert s.factor(P - (-O * (bp + 2 * b / r) + 3 * b * Op) / L) == 0
# Physical scalar curvature of Omega^-2 delta and vacuum Hamiltonian.
Rphys = 4 * O * (Opp + 2 * Op / r) - 6 * Op**2
H = Rphys + 4 * (O * bbp - b2 * Op) / (r * L) + 2 * b2 / r**2
assert s.factor(H) == 0
# Radial physical momentum covector: -2Kt' + 2(R'/R)(Kr-Kt).
M = 2 * bp / r - 2 * b / r**2 + 2 * L / (r * O) * (kr + b / r)
assert s.factor(M) == 0
# Root's d=wF and prior e=exp(g) factor are exactly the same.
root_factor = w * r * ((1 - w) * gp * D + E)
e = s.symbols('e', positive=True)
old_factor = r * e * (D * gp / (1 + e)**2 + E / (1 + e))
assert s.factor(root_factor.subs(w, e / (1 + e)) - old_factor) == 0

receipt = {
    'status': 'PASS_READ_ONLY_MATHEMATICAL_REVIEW',
    'reviewed_index_sha256': wanted,
    'verified_frozen_files': len(catalog['files']),
    'sympy': s.__version__, 'source_sha256': sha(Path(__file__)),
    'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'identities': ['Penrose radial metric=1', 'ADM alpha=L',
                   'physical Kt=-b/r', 'general physical trace',
                   'vacuum Hamiltonian=0', 'vacuum radial momentum=0',
                   'root cutoff factor equals existing p1 log-boost factor'],
    'conditions': '0<r0<r1<S and a>=S/2; Omega>0 on PDE domain',
    'scope': 'Symbolic geometry/endpoint review of existing exponent-one family; no kernel/native/global/stability gate',
    'seconds': time.monotonic() - start,
}
(HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
print(json.dumps(receipt, indent=2))
