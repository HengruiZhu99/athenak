#!/usr/bin/env python3
"""One-shot compact review index; no science or external runtime payload access."""
import hashlib
import json
import pathlib

base = pathlib.Path(__file__).resolve().parent
assert not (base / 'index.json').exists()
files = {}
for path in sorted(base.rglob('*')):
    if not path.is_file():
        continue
    data = path.read_bytes()
    data.decode('utf-8')
    assert len(data) <= 1048576
    files[str(path.relative_to(base))] = {
        'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
receipt = json.loads((base / 'receipt.json').read_text())
addendum = json.loads((base / 'parent-link-addendum.json').read_text())
root = base.parents[2]
for path, expected in receipt['pins'].items():
    if path == 'docs/hyperboloidal-reference-wave-map-native-audit.md':
        assert expected == addendum['original_reviewed_document_sha256']
        expected = addendum['parent_final_document_sha256']
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected
index = {
    'scope': 'Immutable saved-only independent native-preflight document review; no arrays or t2 outputs.',
    'passed': True, 'files': files, 'file_count_excluding_index': len(files),
    'bytes_excluding_index': sum(record['bytes'] for record in files.values()),
    'reviewed_native_doc_sha256': receipt['pins']['docs/hyperboloidal-reference-wave-map-native-audit.md'],
    'parent_final_link_only_doc_sha256': addendum['parent_final_document_sha256'],
    'receipt_sha256': files['receipt.json']['sha256'],
    'completed_preflight_catalog_sha256': receipt['archive_sha256'],
    'all_external_reviewed_text_pins_equal_at_finalize': True,
    'no_new_scientific_queries': True, 'no_t2_output_inspection': True,
    'commands': [
        'python3 docs/validation/hyperboloidal-reference-wave-map-native-doc-review-20261009/review_saved.py',
        'python3 docs/validation/hyperboloidal-reference-wave-map-native-doc-review-20261009/finalize_index.py']}
(base / 'index.json').write_text(json.dumps(index, indent=2, allow_nan=False) + '\n')
print(json.dumps({'index_sha256': hashlib.sha256((base / 'index.json').read_bytes()).hexdigest(),
                  'indexed_files': len(files), 'indexed_bytes': index['bytes_excluding_index'],
                  'receipt_sha256': index['receipt_sha256'],
                  'native_doc_sha256': index['reviewed_native_doc_sha256']}, indent=2))
