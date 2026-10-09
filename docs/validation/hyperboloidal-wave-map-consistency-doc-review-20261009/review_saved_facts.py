"""Read saved archive facts and final prose only; no source/API execution."""
from pathlib import Path
import hashlib
import json
import math
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT/'docs/validation/hyperboloidal-wave-map-consistency-experiments-20261009'
DOC = ROOT/'docs/hyperboloidal-wave-map-consistency-audit.md'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def main():
    raw = DOC.read_bytes()
    text = raw.decode()
    assert sha(ARCHIVE/'catalog.json') == (
        '734f3d83f7f971c3bdd3c345e8a269254c29f0ee9eeb6aa59303937ec1927294')
    catalog = load(ARCHIVE/'catalog.json')
    inputs = [ARCHIVE/'catalog.json']
    for spec in catalog['capsules']:
        p = ARCHIVE/spec['label']/'index.json'
        assert sha(p) == spec['sha256']
        inputs.append(p)
        if spec['label'] != 'independent-binder':
            assert spec['sha256'] in text
    count = len(catalog['files'])
    bytes_total = sum(q.stat().st_size for q in ARCHIVE.rglob('*') if q.is_file())
    assert count == 515 and bytes_total == 11560993
    assert len(catalog['omitted_large_payloads']) == 32
    assert len(catalog['external_originals_rehashed']) == 1476
    for phrase in ['515 copied files', '11,560,993 bytes', '32 payloads',
                   '135 finite JSON files', '1,476 external originals']:
        assert phrase in text
    principal = ARCHIVE/'principal-core'
    idx = load(principal/'index.json')
    attempt = principal/idx['accepted_attempt']
    pm = load(attempt/'check-principal-release.json')
    cm = load(attempt/'core-release.json')
    oi = load(ARCHIVE/'nonlinear-oracle/attempt002/receipt.json')
    bs = load(ARCHIVE/'bounded-coordinate/final/summary.json')
    ai = load(ARCHIVE/'actual-rhs/index.json')
    ap = ARCHIVE/'actual-rhs'/ai['accepted_attempt']
    ar = load(ap/'receipt.json')
    analysis = load(ap/'analysis-release/receipt.json')
    root_review = ROOT/'docs/validation/hyperboloidal-wave-map-consistency-compact-review-20261009/receipt.json'
    assert sha(root_review) == '9b9a80655f8922e26b4277ee86b3dfb1ab686961e1d5eb93578f211d3f6acbc2'
    inputs += [attempt/'check-principal-release.json', attempt/'core-release.json',
               ARCHIVE/'nonlinear-oracle/attempt002/receipt.json',
               ARCHIVE/'bounded-coordinate/final/summary.json',
               ap/'receipt.json', ap/'analysis-release/receipt.json', root_review,
               ARCHIVE/'actual-rhs/probe.cpp', ARCHIVE/'principal-core/core.cpp',
               ARCHIVE/'principal-core/principal.cpp',
               ARCHIVE/'independent-principal-core/REPORT.md',
               ARCHIVE/'addenda/CORE-LINEAR-SCOPE-ADDENDUM.md']
    assert pm['actual_principal_cases'] == 3*3*2*4*11 == 792
    assert cm['witnesses'] == 17 and cm['core_cases'] == 17*4*3 == 204
    assert oi['cases'] == 96 and oi['checks'] == 25152
    assert oi['precision_comparisons'] == 62004 and oi['native_reference_checks'] == 3840
    assert bs['coordinate_cases'] == 629 and bs['binding_cases'] == 420
    assert bs['all629_scientific_common_fields_equal_exactly']
    assert len(ar['source_before']) == 381 and len(ar['commands']) == 5
    assert analysis['cases'] == 48 and ar['release_debug_equal']
    facts = [
        ('bounded geometry', bs['coordinate_maxima']['geometry_entry_scaled'], '3.0013e-11'),
        ('bounded final FD', bs['coordinate_maxima']['fd_final_scaled'], '7.9702e-8'),
        ('original failed embedding', bs['original_uncanceled_direct_identity_max'], '1.1209e-8'),
        ('negative corruption', bs['negative_returned_jet_control_min'], '7.583e-8'),
        ('principal matrix', pm['matrix_expected_max_absolute'], '4.0412e-13'),
        ('principal involution', pm['actual_involution_max_absolute'], '4.0867e-13'),
        ('principal symmetrizer', pm['actual_symmetrizer_max_absolute'], '3.2219e-13'),
        ('principal normals', pm['normal_max_scaled'], '2.6277e-16'),
        ('linear core RHS', cm['all22_max_absolute'], '1.1103e-15'),
        ('linear core constraints', cm['physical8_initial_constraint_max'], '6.6614e-16'),
        ('linear core acceleration', cm['coordinate_acceleration_max_absolute'], '3.5528e-15'),
        ('oracle analytic', oi['analytic_scaled_max'], '4.0266e-72'),
        ('oracle precision', oi['precision_scaled_max'], '5.4335e-71'),
        ('RHS22', analysis['maxima']['rhs22_scaled'], '2.4159e-13'),
        ('connection', analysis['maxima']['connection_scaled'], '4.2244e-16'),
        ('scaled source', analysis['maxima']['source_scaled'], '2.4981e-15'),
        ('physical8', analysis['maxima']['physical_constraints_absolute'], '1.3781e-14'),
        ('input normals', analysis['maxima']['input_normals_absolute'], '3.3307e-16'),
        ('output normals', analysis['maxima']['rate_normals_scaled'], '4.1237e-14'),
        ('Omega jets', analysis['maxima']['omega_difference_absolute'], '8.8818e-16'),
    ]
    # This tolerance is for prose decimal rounding, not a scientific gate.
    for name, actual, printed in facts:
        assert printed in text, name
        assert abs(actual-float(printed))/abs(actual) < 1e-4, name
    for mode, expected_count in [('release', 1049), ('debug', 1051)]:
        b = load(ap/('build-'+mode+'.json'))
        inputs.append(ap/('build-'+mode+'.json'))
        assert b['compiler_dependencies_count'] == expected_count
        assert b['executable_sha256'] in text
    for expected in [ar['recipe_sha256'], ar['execution_release_sha256']]:
        assert len(expected) == 64
    branch = subprocess.check_output(['git','rev-parse','--abbrev-ref','HEAD'],cwd=ROOT,text=True).strip()
    assert branch == 'z4c_hyperboloidal_layer'
    diff = subprocess.check_output(['git','diff','--name-only',
             '27c19d20696ea6dd4704032c51dfd026218f64f2','--','src','CMakeLists.txt'],cwd=ROOT,text=True)
    assert diff == ''
    assert 'The reference retains its exact Cauchy core' in text
    assert 'runtime kappa input 10' in text and 'damping argument `10/alpha`' in text
    assert 'prior physical-P/spatial-norm gauge' in text
    (HERE/'reviewed-audit.md').write_bytes(raw)
    result = {'passed': True, 'scope': 'independent saved-only factual/source-convention/prose review; no new scientific queries',
              'document_sha256': hashlib.sha256(raw).hexdigest(),
              'review_source_sha256': sha(__file__),
              'catalog_sha256': sha(ARCHIVE/'catalog.json'),
              'inputs': {str(p.relative_to(ROOT)): sha(p) for p in inputs},
              'saved_numeric_facts': [{'name': n, 'exact_saved_value': v, 'printed': s} for n,v,s in facts],
              'counts': {'principal_matrices': 792, 'linear_core': 204, 'bounded_coordinate': 629,
                         'bounded_binding': 420, 'oracle_precision_cases': 96, 'distinct_inputs': 48,
                         'oracle_checks': 25152, 'native_reference_subset': 3840,
                         'precision_comparisons': 62004, 'copied_files_excluding_catalog': 515,
                         'total_archive_bytes_including_catalog': 11560993,
                         'finite_JSON_including_catalog': 135, 'metadata_only_payloads': 32,
                         'external_originals': 1476},
              'production_src_and_root_CMake_unchanged': True,
              'applied_nonblocking_clarifications': [
                  'Reference core remains exact Cauchy; witness perturbation fields need not remain the reference.',
                  'Runtime damping input10 is distinct from the ConformalRHS argument10/alpha.',
                  'Bounded coordinate gate uses the prior physical-P/spatial-norm gauge; it is not a wave-map evolution gate.'
              ],
              'source_and_scope_findings': [
                  'Principal proof is exact for the analytic normalized expected matrix; actual792 floating matrices are a separate consistency check. No common3D, puncture or scri estimate is inferred.',
                  'Core204 amplitude-dual cases are linearized, and actual aggregate exports do not supply per-case residuals. Origin scalar limit is analytic and properly distinguished.',
                  'Nonlinear oracle uses inverse physical-inertial shear with third map jets and second physical metric jets. Source conventions physicalP=K-2Theta, Theta_phys, Cartesian contravariant beta and Omega*Fbar are preserved.',
                  'Bounded subsection Falpha/Fbeta denote lapse/shift RHS rates, distinct from Fbar GH source later. Gaussian F(s) is separately defined in the oracle construction.',
                  'All48 raw22 point actions use one assembly and no reference RHS subtraction/projection. Native code retains its separate geometric roundoff subtraction; document flags this distinction for future preflight.',
                  'Exact flat sampled consistency does not establish lower-order, native, exact-scri or black-hole stability. Later wormhole-to-trumpet survival with Minkowski reference is explicitly retained.',
                  'Archive count/hash/metadata and empty frozen stdout versus later additive logs match the frozen collection.'
              ], 'remaining_corrections': [],
              'frozen_archive_modified': False}
    (HERE/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed': True, 'document_sha256':result['document_sha256'],
                      'receipt_sha256':sha(HERE/'receipt.json'), 'numeric_facts_checked':len(facts),
                      'remaining_corrections': []},indent=2))


if __name__ == '__main__':
    main()
