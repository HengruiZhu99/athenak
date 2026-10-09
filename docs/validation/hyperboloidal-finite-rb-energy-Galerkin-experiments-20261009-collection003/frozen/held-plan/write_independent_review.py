"""Read-only hash and mathematical advisory review; no radial computation."""
import hashlib
import json
import pathlib
import subprocess

root = pathlib.Path('/Users/hz0693/research/hyperboloidal')
base = root / 'build-layer-research/boundary/total-j-finite-rb-control-held-20261009'
expected = {
    'RECIPE.md': 'f7e1cc7bd1d7dc5a466dd66af56f7ca41dd41f758689e6a37e570f8fdd15ae89',
    'preparation-receipt.json': '5a6ee19caf900a1201347bee697fcd6704f4b829405dceef4fa51774462830a4',
    'source-pins.json': '3bb48f6a36bdcc702ba2b128657dfeb0cfbfc76ea349d5525294b10ef9571aa6',
    'FINAL-ADDENDUM.md': '69c13c353efd91f08315578f5c7dfaa6413dfed8be36262c1df22eb7dffab163',
    'final-addendum-receipt.json': 'ca33786fb098622e5775f7a9af5bfdf07f7a1854faf16790011aa3d0853b265e',
}


def pin(path, digest):
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    assert actual == digest, (str(path), actual, digest)
    return {'path': str(path), 'sha256': actual, 'bytes': len(data)}


reviewed = {name: pin(base / name, digest) for name, digest in expected.items()}
source_pins = json.loads((base / 'source-pins.json').read_text())
verified = {name: pin(pathlib.Path(row['path']), row['sha256'])
            for name, row in source_pins.items()}
addendum = json.loads((base / 'final-addendum-receipt.json').read_text())
for name in ('subsidiary_manifest', 'subsidiary_helper'):
    row = addendum[name]
    verified[name] = pin(pathlib.Path(row['path']), row['sha256'])
review = {
    'status': 'PASS_read_only_math_source_advisory_no_blocking_corrections',
    'reviewer': '/root/literature_gauge',
    'review_time_utc': '2026-10-09 13:56:56 UTC',
    'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                    text=True).strip(),
    'method': 'Independent source/formula review and SHA256 readback only.',
    'scientific_compile': False,
    'actual_radial_operator_or_matrix': False,
    'kernel_or_scalar_model_rerun': False,
    'spectrum_propagation_or_evolution': False,
    'reviewed_sources': reviewed,
    'verified_input_pins': verified,
    'findings': [
        'Energy Galerkin E^-1 K is the declared H1-type Riesz projection of the complete actual point action. It is not strong collocation and changes radial bulk as well as boundary, so it cannot isolate native ghosts.',
        'The full normalized configuration derivative q=Ds U retains reference/frame coefficient derivatives. The exact five-component tensor chart and its inverse, contravariant Lambda coframe, and A trace-tangent subtraction including the configuration RHS are explicit and consistent.',
        'Strong q_t differentiates first-order configuration equations only. Second configuration/reference jets and first A jets suffice; no differentiation of second-order momentum RHS or invented third reference jets is admitted.',
        'dV=r^2 L/alpha_hat dr dOmega, Ds=(alpha_hat/L) partial_r, div(s)=2 alpha_hat/(r L), dSigma=rb^2 dOmega and B=rb sqrt(w_angle) times the complete trace agree. Grouped origin density and exact polynomial core avoid evaluating the singular normal frame at the origin.',
        'The H_C/H_D blend is confined to W=1, where both symmetrize the same normal matrix. The explicit Gamma=Ds(H Kn)+(div s)H Kn and independent remainder formula give the stated symmetric G_volume with the correct negative coefficient-gradient sign.',
        'The independent weak/strong and point-source G_volume identity is non-tautological because neither the remainder nor G_volume is defined as an assembled matrix residual. Angular, reference, A-trace and gauge terms remain included.',
        'The adjoint SAT has the correct sign for RHS y_t=Kn Ds y. With E-energy one half, it changes the incoming boundary flux to -kin/2 times the incoming squared H-norm while retaining kout/2 outgoing flux. It updates both configuration and momentum and preserves the identity q=Ds U.',
        'Expected fixed-J incoming ranks 4/8/10 and splits 2+2+0, 4+4+0, 4+4+2 must be measured on the restricted trace and sector maps; they are not already established by full20 counts. The addendum explicitly requires this gate.',
        'Actual-source manufactured forcing formed before assembly checks assembly consistency. Separate exact core, derivative, raw22 normal and coefficient-aware subsidiary checks provide independent continuum-action evidence. The FD transition/collar rates retain all levels, convergence/floor evidence and absolute errors instead of fabricating higher reference jets.',
        'The frozen Subsidiary helper uses physical H, M covector, Z covector, Theta and complete C0 stationary-Einstein reference coefficients. Projection and SAT constraint effects are correctly reported separately; this principal SAT is not CPBC.',
        'The retained recipe negative incoming speed numbers are propagation speeds -kin; the RHS eigenvalue kin itself is positive. The equations and SAT sign are consistent with this convention.',
        'The condition cap and backward solve residual do not certify forward spectral accuracy. Finite quadrature/matrix identities, trace conditioning, J<=2 and finite rb remain limited controls, not continuum, exact-scri, nonlinear or degree-uniform stability.'
    ],
    'blocking_corrections': [],
    'scope': {
        'advisory_admissible_next_stage': 'Only parent-authorized source/configuration-derivative/mass/operator N8 rb=.98 J=0/1/2 gates, preserving failed evidence and declared thresholds.',
        'no_solver_release_by_this_receipt': True,
        'excluded': ['eigenvalues', 'propagation', 'native evolution', 'CPBC',
                     'exact scri closure', 'J>=3', 'nonlinear pulse acceptance',
                     'BH integration or evolution'],
        'eventual_user_target': 'Substantial angular/nonradial Minkowski pulse acceptance first; later a single BH must survive wormhole-to-trumpet interior transition with the Minkowski hyperboloidal reference throughout.'
    },
    'review_source': pin(pathlib.Path(__file__), hashlib.sha256(
        pathlib.Path(__file__).read_bytes()).hexdigest()),
}
output = base / 'independent-advisory-review.json'
assert not output.exists(), 'Preserve accepted review bytes; use a fresh path.'
output.write_text(json.dumps(review, indent=2, allow_nan=False) + '\n')
print(json.dumps({'path': str(output), 'sha256': hashlib.sha256(
    output.read_bytes()).hexdigest(), 'bytes': output.stat().st_size,
    'reviewed_file_count': len(reviewed), 'verified_source_pins': len(verified),
    'status': review['status']}, indent=2))
