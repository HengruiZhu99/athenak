"""Create a separate N20 read-only search with otherwise identical method."""
from pathlib import Path
import hashlib
import json

P = Path(__file__).resolve().parent
source = (P/'reduced_modes.py').read_text()
changes = {
    "OLD = B/'full-tensor-propagator/full22-v2'": "OLD = B/'full-tensor-C0-N20-20261009/full22'",
    "LONG = B/'full-tensor-C0-long-window-20261009'": "LONG = B/'full-tensor-C0-N20-20261009/full22'",
    "FROZEN = LONG/'immutable-C0-long-window-20261009'": "FROZEN = B/'full-tensor-C0-N20-20261009/immutable-C0-N20-default-span-v2-20261009'",
    "'d1efcbea11e730eec2e784ef7981e64f5e67d64e51bc963c6cdd7ed22834d896'": "'8ec4c4dd84898d19b055831696de6e3608542b1fd10a88e04f087b1f689f99aa'",
    "'large-output-metadata.json'": "'large-artifacts-metadata-only.json'",
    "'767b7c998e27db4d598e260f80e2181afe31d427c3f58c9a7bb60e30c35dede6'": "'bfc114a9495b49d01b9275f4efd8a1fe49d167273f8d849631d25cddf9c2a103'",
    "'b02bf12087248a96d1b217d164e6d960cfc506791892b1058d7c2df018ed82cf'": "'2e777543e187896882e7a97f4f23a63fee057145a8c4f034932a782ebe0154c7'",
    "assert J.shape == (32800, 32800) and values.shape == (241, 32800, 2)": "assert J.shape[0] == J.shape[1] == values.shape[1] and values.shape[0] == 241 and values.shape[2] == 1",
}
for a, b in changes.items():
    assert a in source, a
    source = source.replace(a, b)
start = source.index('cases = {')
end = source.index('\nassert args.case in cases', start)
oldcases = source[start:end]
newcases = """cases = {
    'n20-late2-gauge-s2': (2., 6., [0], 2, True),
    'n20-late4-gauge-s1': (4., 6., [0], 1, True),
    'n20-late4-gauge-s2': (4., 6., [0], 2, True),
}"""
source = source[:start]+newcases+source[end:]
source = source.replace('C0 N16', 'C0 N20')
(P/'n20_reduced_modes.py').write_text(source)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {'scope': 'N20 phase-confounded control, method unchanged except pinned inputs and one-seed cases',
          'original_source_sha256': sha(P/'reduced_modes.py'),
          'generated_source_sha256': sha(P/'n20_reduced_modes.py'),
          'literal_changes': changes, 'old_cases': oldcases, 'new_cases': newcases,
          'scope_label_change': 'C0 N16 to C0 N20'}
(P/'n20-preparation.json').write_text(json.dumps(record, indent=2)+'\n')
print('PREPARED', record['generated_source_sha256'])
