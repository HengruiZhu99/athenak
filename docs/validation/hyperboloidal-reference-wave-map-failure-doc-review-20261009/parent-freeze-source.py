"""Preserve the independent review and verify the exact parent text correction."""
from pathlib import Path
import difflib
import hashlib
import json

ROOT = Path('/Users/hz0693/research/hyperboloidal')
SOURCE = ROOT / 'build-layer-research/wave-map-failure-doc-independent-review-20261009'
DEST = ROOT / 'docs/validation/hyperboloidal-reference-wave-map-failure-doc-review-20261009'
DOC = ROOT / 'docs/hyperboloidal-reference-wave-map-failure-audit.md'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert not DEST.exists()
    assert sha(SOURCE / 'index.json') == '5ed807af71c7c05bc2ccebf003b5aae4d32970340c7575b784bc90f26e6c2f23'
    before = {str(p.relative_to(SOURCE)): {'source': str(p), 'bytes': p.stat().st_size,
                                          'sha256': sha(p)}
              for p in SOURCE.rglob('*') if p.is_file()}
    index = json.loads((SOURCE / 'index.json').read_text())
    for rel, item in index['files'].items():
        assert before[rel]['bytes'] == item['bytes'] and before[rel]['sha256'] == item['sha256']
    original = (SOURCE / 'reviewed-document.md').read_text()
    link = '''
The [independent checkpoint review](validation/hyperboloidal-reference-wave-map-failure-doc-review-20261009/README.md)
verified copied originals, all 167 scalar observations and conditional math
scopes. It flagged the N24 shell-fraction rounding typo, corrected above; its
original reviewed document and the exact parent correction are retained.
'''
    assert original.count('.9835844') == 1
    expected = original.replace('.9835844', '.9835841') + link
    assert DOC.read_text() == expected, 'Unexpected additional document change'
    scalar_path = (ROOT / 'build-layer-research/reference-wave-map-partial-N24-held-20261009'
                   '/attempts/wave-map-N24-large-t2-001/snapshot-observations.json')
    scalar_sha = sha(scalar_path)
    assert scalar_sha == '165fece3367d9af994596744e394b33904f71cd4ca03a9dde3224266586de0e4'
    scalars = json.loads(scalar_path.read_text())
    fraction = sum(v['squared_fraction4'][2] for v in scalars[-1]['native_diagnostics']['radial_bins']
                   if v['rlo'] >= .9)
    assert format(fraction, '.7f') == '0.9835841'
    final_doc_sha = sha(DOC)
    DEST.mkdir(parents=True)
    for rel, item in before.items():
        p, q = Path(item['source']), DEST / rel
        assert p.stat().st_size <= 1024 * 1024
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(p.read_bytes())
        assert sha(q) == item['sha256']
    diff = ''.join(difflib.unified_diff(original.splitlines(True), expected.splitlines(True),
                                        fromfile='reviewed-original.md', tofile='final-document.md'))
    (DEST / 'parent-correction.diff').write_text(diff)
    (DEST / 'parent-freeze-source.py').write_bytes(Path(__file__).read_bytes())
    receipt = {'passed_parent_correction_check': True,
               'original_review_disposition': 'completed_with_one_documentary_correction_required',
               'original_doc_sha256': sha(SOURCE / 'reviewed-document.md'),
               'final_doc_sha256': final_doc_sha,
               'independent_review_index_sha256': sha(SOURCE / 'index.json'),
               'actual_N24_squared_Z_fraction_r_ge_09': fraction,
               'saved_scalar_path': str(scalar_path), 'saved_scalar_sha256': scalar_sha,
               'seven_place_rounding': '.9835841',
               'permitted_changes_only': ['one rounding correction', 'independent review link paragraph'],
               'diff_sha256': sha(DEST / 'parent-correction.diff'),
               'source_sha256': sha(__file__), 'new_scientific_queries': 0, 'new_native_steps': 0}
    dump(DEST / 'parent-correction-receipt.json', receipt)
    dump(DEST / 'copy-catalog.json', before)
    (DEST / 'README.md').write_text('''# Independent failed-pulse checkpoint review

The independent review is preserved byte-for-byte, including its original
document and explicit required N24 shell-fraction rounding correction.
Its disposition is not retroactively changed to PASS. The parent corrected
.9835844 to .9835841, checked against the unchanged saved scalar data, and added
the review link. The exact diff and final document SHA are recorded separately.
No other document content changed. All copied review sources, commands, logs,
receipts and input indexes remain unchanged. No raw arrays or scientific/native
calls were needed for this preservation and text correction check.
''')
    for rel, item in before.items():
        assert sha(item['source']) == item['sha256'], rel
    assert sha(DOC) == final_doc_sha
    assert sha(scalar_path) == scalar_sha
    print(json.dumps(receipt, allow_nan=False))


if __name__ == '__main__':
    main()
