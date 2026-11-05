"""Native map oracle and interpolation checks, without new physics evolution."""
from pathlib import Path
import json
import subprocess
import numpy as np
from sample_rf2_fields import Checkpoint, Geometry, weights, sample, digest

D = Path(__file__).resolve().parent
R = next(p for p in D.parents if (p/'CMakeLists.txt').exists())
O = R/'output/refined-convergence/common-field-sampling-check'
E = D/'common-field-sampling-check'; E.mkdir(exist_ok=True)
v = json.loads((D/'sealed-topology-controls/verification.json').read_text())['full_smr_checkpoint']
path = R/'output/refined-convergence/sealed-topology-candidate-v1/physics/full-smr-radial24-angular2.checkpoint'
config = R/'inputs/hyperbolic_shift_3d/refined_convergence/central20_boosted_smr_tube_radial24_angular2_reference.cfg'
cp = Checkpoint(path, sha256=v['sha256'], identity=v['identity'])
assert cp.n == v['n'] and cp.steps == v['steps'] and cp.time == v['time']
g = Geometry(cp.leaves, cp.subdivisions, config)
assert len(g.blocks) == 832 and g.core_count == 400
np.savetxt(O/'leaves.txt', cp.leaves, fmt='%d')
rng = np.random.default_rng(60412)
points = list(rng.uniform(-.399, .399, size=(256, 3)))
directions = rng.normal(size=(1024, 3)); directions /= np.linalg.norm(directions, axis=1)[:, None]
points.extend(directions*rng.uniform(.70, .99, size=(1024, 1)))
for radius in [.02, .05, .1, .2, .35, .4, .54, .78, .875, .9, .925, .95, .99]:
    for axis in range(3):
        for sign in [-1, 1]:
            x = np.zeros(3); x[axis] = radius*sign; points.append(x)
points = np.array(points)
np.savetxt(O/'points.txt', points, fmt='%.17g')
cmd = [str(O/'oracle'), '12', str(config), str(O/'leaves.txt'), str(O/'points.txt')]
result = subprocess.check_output(cmd, text=True)
(O/'native-oracle.txt').write_text(result)
blocks = []; samples = []
for line in result.splitlines():
    s = line.split()
    if s[0] == 'B': blocks.append(s)
    elif s[0] == 'P': samples.append(s)
assert len(blocks) == len(g.blocks) and len(samples) == len(points)
block_error = q_error = roundtrip_error = 0.
for s, b in zip(blocks, g.blocks):
    assert tuple(map(int, s[2:5])) == b[:3]
    block_error = max(block_error, float(np.max(np.abs(np.array(s[5:], float)-np.r_[b[3], b[4]]))))
for s, x in zip(samples, points):
    owner, q = g.owner(x)
    assert owner == int(s[2]), (x, owner, int(s[2]))
    q_error = max(q_error, float(np.max(np.abs(q-np.array(s[3:6], float)))))
    roundtrip_error = max(roundtrip_error, float(np.max(np.abs(x-np.array(s[6:9], float)))))
assert block_error < 2e-15 and q_error < 2e-12 and roundtrip_error < 2e-15

# Reproduce polynomials through each stencil's degree, including block edges.
polynomial_errors = {}
for stencil in [6, 8]:
    maximum = 0.
    for n in [12, 14, 16]:
        for q in np.linspace(0, 1, 61):
            indices, w = weights(q, n, stencil)
            for degree in range(stencil):
                expected = q**degree
                actual = np.dot(w, ((indices+.5)/n)**degree)
                maximum = max(maximum, abs(actual-expected))
    assert maximum < 5e-13
    polynomial_errors[str(stencil)] = maximum

# At actual stored nodes, the mapping/interpolation must recover all31 values.
node_points, expected = [], []
nominal = len(g.blocks)*cp.n**3
data = cp.arrays[0][:31*nominal].reshape(len(g.blocks), cp.n, cp.n, cp.n, 31)
for m in rng.permutation(len(g.blocks)):
    face, layer, level, lo, hi = g.blocks[m]
    i, j, k = rng.integers(1, cp.n-1, size=3)
    local = (np.array([i, j, k])+.5)/cp.n; q = lo+local*(hi-lo)
    if face < 0:
        x = q
    else:
        a = face//2; lower = g.inner if layer == 0 else g.radii[layer-1]
        radius = lower+q[0]*(g.radii[layer]-lower)
        uv = np.tan(np.pi*q[1:]/4); scale = radius/np.sqrt(1+np.dot(uv, uv))
        x = np.zeros(3); x[a] = (-1 if face % 2 else 1)*scale
        x[(a+1)%3] = scale*uv[0]; x[(a+2)%3] = scale*uv[1]
    if np.linalg.norm(x) < .08 or g.owner(x)[0] != m:
        continue
    node_points.append(x); expected.append(data[m, k, j, i].copy())
    if len(node_points) == 128: break
node_errors = {}
for stencil in [6, 8]:
    sampled, _ = sample(cp, g, node_points, stencil)
    expected_array = np.array(expected)
    error = float(np.max(abs(sampled-expected_array)/(1+abs(expected_array))))
    assert error < 2e-12
    node_errors[str(stencil)] = error
assert digest(path) == v['sha256']
report = dict(passed=True, checkpoint_sha256=v['sha256'], checkpoint_unchanged=True,
              configuration_sha256=digest(config), native_oracle_command=cmd,
              native_oracle_binary_sha256=digest(O/'oracle'), blocks=len(blocks),
              common_points=len(points), all_native_owners_equal=True,
              maximum_block_bound_error=block_error, maximum_inverse_map_error=q_error,
              maximum_native_roundtrip_error=roundtrip_error,
              polynomial_reproduction_errors=polynomial_errors,
              stored_nodes_checked=len(node_points), stored_node_scaled_errors=node_errors,
              sampler_source_sha256=digest(D/'sample_rf2_fields.py'),
              qualification='Analysis-only map and interpolation validation, not boosted evolution or a convergence result. Common-coordinate field sampling must compare both interpolation orders; native constraint norms remain separate. No input checkpoint or production source changed.')
(E/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
(E/'build.json').write_bytes((O/'build.json').read_bytes())
print(json.dumps(report, indent=2))
