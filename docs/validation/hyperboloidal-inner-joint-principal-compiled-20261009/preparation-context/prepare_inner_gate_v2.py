"""Standard-library source preparation only. Generated gates are not executed."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
OLD = P / 'inner-joint-principal-gate-held-20261009'
NEW = P / 'inner-joint-principal-gate-v2-held-20261009'
if NEW.exists():
    raise FileExistsError(NEW)
NEW.mkdir()
for name in ('probe.cpp', 'gauge_proposal.hpp', 'check_exact.py',
             'reference_wave_map.hpp', 'PLAN.md'):
    shutil.copyfile(OLD / name, NEW / name)
probe = (NEW / 'probe.cpp').read_text()
assert probe.count('count!=262') == 1
(NEW / 'probe.cpp').write_text(probe.replace('count!=262', 'count!=118'))
plan = (NEW / 'PLAN.md').read_text()
assert plan.count('grid is216 ordinary') == 1 and plan.count('Total262') == 1
plan = plan.replace('grid is216 ordinary', 'grid is72 ordinary').replace('Total262', 'Total118')
(NEW / 'PLAN.md').write_text(plan)
parts = []
for name in ('probe.cpp', 'PLAN.md'):
    parts.extend(difflib.unified_diff((OLD / name).read_text().splitlines(True),
                 (NEW / name).read_text().splitlines(True),
                 fromfile=str(OLD / name), tofile=str(NEW / name)))
(NEW / 'count-only.diff').write_text(''.join(parts))
(NEW / 'COUNT-CORRECTION.md').write_text('''# Source-preparation count correction

The first held proposal is preserved byte for byte. Its ordinary fixed grid has
3 alpha values x2 chi values x3 W values x2 G0 values x2 frames =72 cases.
Adding8 mu1 cases,2 q=f cases and36 general cases yields118, rather than262.
The original source-only proposal therefore failed its own count prerequisite;
no gate import, compile, exact-algebra run or kernel query had occurred.

The exact count-only diff changes the final C++ count assertion and two prose
counts. The probe points, source formulas, exact checker and thresholds remain
unchanged. The new runner/analyzer/recipe are separately prepared for root
review and are not an authorization to execute.
''')
for name in ('check_exact.py',):
    ast.parse((NEW / name).read_text(), filename=str(NEW / name))
print(json.dumps({'new': str(NEW), 'scientific_execution': False,
                  'old_index_sha256': hashlib.sha256((OLD / 'source-index.json').read_bytes()).hexdigest()}, indent=2))
