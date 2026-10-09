"""Read-only hash/source/saved-result review; no new scientific execution."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
gates = [
    ('core', 'build-layer-research/boundary/total-j-flat-core-envelope-20261009/'
     'immutable-total-J-flat-core-envelope-20261009',
     'b0fde1e0eb95d6660a9fa3d190eda69207e6153c88da35366b038369ac3aa3d4'),
    ('principal', 'build-layer-research/continuum/harmonic-principal-constraint-sectors/'
     'immutable-harmonic-normal-principal-sectors-20261009',
     '05d4d7308477efe26d26ec7256fd9ba824040849363ed7f723031e5760adc0fc')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


checked = []
for name, directory, pin in gates:
    base = ROOT/directory
    assert sha(base/'index.json') == pin
    index = json.loads((base/'index.json').read_text())
    for row in index['files']:
        path = base/row['path']
        assert sha(path) == row['sha256']
        assert path.stat().st_size == row['bytes']
    external = index.get('large_files_metadata_only',
                         index.get('large_inputs_metadata_only', []))
    for row in external:
        path = Path(row.get('external_path', ROOT/row.get('original_path', '')))
        assert sha(path) == row['sha256']
        assert path.stat().st_size == row['bytes']
    checked.append({'name': name, 'index': directory+'/index.json',
                    'sha256': pin, 'frozen_files_verified': len(index['files']),
                    'external_metadata_records_verified': len(external)})

core = ROOT/gates[0][1]
symbolic = json.loads((core/'symbolic-report.json').read_text())
local = json.loads((core/'local-report.json').read_text())
cartesian = json.loads((core/'cartesian-oracle-report.json').read_text())
assert symbolic['sparse_polynomial_entries'] == 188
assert symbolic['maximum_rho_degree'] == 2
assert symbolic['counts']['exact_all_m_independent_jet_actions'] == 468
assert symbolic['all_exact_cartesian_residuals_zero']
assert not symbolic['r_or_rho_denominators']
assert not symbolic['imposed_cross_L_envelope_conditions']
assert local['rows'] == 19500 and all(local['checks'].values())
assert cartesian['release'] == cartesian['debug']
assert cartesian['release']['cases'] == 2800
blocks = local['saved_fitted_core_comparisons']
assert len(blocks) == 18
worst_scaled = max(blocks, key=lambda row: row['scaled'])
worst_entry = max(blocks, key=lambda row: row['absolute_max'])
assert worst_scaled['J'] == 2 and worst_scaled['derivative'] == 1
assert worst_entry['J'] == 2 and worst_entry['derivative'] == 0
principal = ROOT/gates[1][1]
report = json.loads((principal/'report.json').read_text())
assert report['constraint_rank'] == 8 and report['combined_left_rank'] == 20
assert report['actual_retained_harmonic_rows'] == 288
for row in report['sectors'].values():
    assert row['projector_rank'] == 10 and row['constraint_image_rank'] == 4
    assert row['coordinate_gauge_rank'] == 4 and row['screen_TT_rank'] == 2

reviewed = [HERE/'audit-draft.md', HERE/'archive-README.md', Path(__file__)]
receipt = {
    'passed_source_math_hash_and_draft_review': True,
    'HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                    cwd=ROOT, text=True).strip(),
    'frozen_inputs': checked,
    'draft_source_pins': {str(path.relative_to(ROOT)): sha(path)
                          for path in reviewed},
    'core_source_review': 'Independent read-only comparison with pinned '
    'FlatFormula: tau/chi normalization, negative-K Hessians/STF projection, '
    'physical-P lapse3,shift3/8 and kappa10 damping factors are consistent. '
    'Exact homogeneous projection plus all-m Cartesian reconstruction '
    'preserves arbitrary independent regular envelopes without divisions.',
    'saved_result_counts': {'exact_jet_actions': 468, 'compiled_queries': 19500,
                           'held_out_Cartesian_cases_per_build': 2800,
                           'fitted_core_blocks': 18,
                           'retained_harmonic_records': 288},
    'historical_REPORT_wording_correction': {
        'largest_absolute_fitted_entry_error': worst_entry['absolute_max'],
        'location': {'J': worst_entry['J'], 'derivative': worst_entry['derivative'],
                     'r': worst_entry['r']},
        'worst_scaled_block_error': worst_scaled['scaled'],
        'entry_error_in_worst_scaled_block': worst_scaled['absolute_max'],
        'action': 'New draft distinguishes these values; historical frozen '
        'REPORT/results/tolerances remain unchanged.'},
    'principal_source_review': 'Exact physical diagnostic map and independent '
    '4D Lie-coordinate/negative-K columns prove4constraint+4gauge+2TT per sign. '
    'Contravariant xi^tau notation, Omega weights, derivative orders, '
    'nonzero-k inverse derivative and nonorthogonal/degenerate sectors are '
    'explicit. No local CPBC follows from this normal principal algebra.',
    'draft_corrections_remaining': [],
    'interpretation': 'Local core action and finite-positive-Omega normal '
    'principal algebra only; no radial discretization/global matrix, boundary '
    'adoption, eigenproblem, evolution, stability or black-hole acceptance.',
    'required_later_BH_transition': 'Wormhole-to-trumpet inner transition '
    'with Minkowski hyperboloidal reference retained throughout.',
    'scientific_rerun_or_collection': False, 'frozen_mutation': False}
(HERE/'source-math-review.json').write_text(
    json.dumps(receipt, indent=2, allow_nan=False)+'\n')
print(json.dumps({'passed': True, 'receipt_sha256':
                  sha(HERE/'source-math-review.json')}, indent=2))
