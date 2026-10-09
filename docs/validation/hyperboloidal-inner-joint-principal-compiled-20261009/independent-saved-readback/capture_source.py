"""Capture completed saved evidence before review; standard-library only."""
from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
DEST = P / 'inner-joint-principal-v3-independent-saved-readback-20261009'
DEST.mkdir(exist_ok=False)
source = P / 'inner-joint-principal-gate-v3-held-20261009'
attempt = source / 'attempts/gate001'
outer = ROOT / 'build-layer-research/inner-joint-principal-v3-root-release-20261009'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if sha(attempt / 'receipt.json') != 'bf778caf62d6596261d5c15a7307b791bb39be321f0bc0bc834adfdeeef4d83b':
    raise ValueError('released completed receipt differs')
(DEST / 'PLAN.md').write_text('''# Independent saved inner-principal readback

Root authorized saved-only review of the completed v3 gate. Capture source,
completed child/outer records and small text outputs before interpreting them.
Never execute a compiler, probe, original analyzer or exact checker. No native,
eigensolve or propagation is allowed.

An independent standard-library checker will verify all118 fixed actual20
matrices against literal complete20 rows at the unchanged2e-12 absolute gate,
directly compose scalar H,V,X,ell rows with the saved matrix twice and test
four-wave identities at the unchanged1e-10 absolute gate. It will check the
saved18 Fraction records algebraically, command/dependency/ASan provenance,
Release/debug byte equality and every saved summary's fixed threshold results.
It will rehash source/header/runtime metadata without executing those inputs.

This is finite positive alpha/chi constant-reference principal evidence only.
No nonlinear source, puncture, black-hole or evolution adoption follows.
All originals and previous failed source-preparation attempts remain unchanged.
''')
files, external = [], []
for label, directory in [('source', source), ('attempt', attempt), ('outer', outer)]:
    paths = sorted(directory.rglob('*')) if label != 'source' else sorted(directory.iterdir())
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(directory)
        raw = path.read_bytes()
        row = dict(original=str(path), relative=str(relative), group=label,
                   sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        try:
            raw.decode('utf-8')
            text = True
        except UnicodeDecodeError:
            text = False
        forbidden = path.suffix in ('.npz', '.npy', '.jsonl', '.o', '.a') or raw[:4] in (
            b'PK\x03\x04', b'\x7fELF', b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf')
        if text and not forbidden and len(raw) <= 1048576:
            copy = DEST / 'inputs-before-review' / label / relative
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, copy)
            row['copy'] = str(copy)
            files.append(row)
        else:
            external.append(row)
(DEST / 'review-inputs.json').write_text(json.dumps(dict(
    saved_only=True, copied=files, metadata_only=external,
    no_compiler_probe_original_gate_import_or_execution=True), indent=2) + '\n')
shutil.copyfile(__file__, DEST / 'capture_source.py')
print(json.dumps(dict(destination=str(DEST), copied=len(files), metadata_only=len(external),
                     inputs_sha256=sha(DEST / 'review-inputs.json')), indent=2))
