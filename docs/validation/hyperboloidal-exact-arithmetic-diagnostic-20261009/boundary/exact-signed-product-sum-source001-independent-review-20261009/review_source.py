"""Stdlib metadata/text source review only; never import candidate or targets."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import shutil
import time

P=Path(__file__).resolve().parent
OWNER=P.parents[1]/'continuum/exact-signed-product-sum-source001-held-20261009'
EXPECTED='0f21b208093a5fe042e7e8f02b300380ac0a88694bf3afee5bbee0b214e72460'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,j):p.write_text(json.dumps(j,indent=2,sort_keys=True)+'\n')

def main():
    t=time.monotonic();assert sha(OWNER/'source-index.json')==EXPECTED
    si=json.loads((OWNER/'source-index.json').read_text());ext=json.loads((OWNER/'external-pins.json').read_text())
    rows=si['files']+ext+[{'path':str(OWNER/'source-index.json'),'sha256':EXPECTED,'bytes':(OWNER/'source-index.json').stat().st_size}]
    pins={}
    for r in rows:
        if r['path']in pins:assert pins[r['path']]==r['sha256']
        pins[r['path']]=r['sha256'];assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256']
    write(P/'inputs-before.json',pins)
    capture=P/'source-copies';capture.mkdir(exist_ok=False)
    for row in si['files']:
        q=Path(row['path']);assert q.stat().st_size<=1<<20;shutil.copyfile(q,capture/q.name)
    shutil.copyfile(OWNER/'source-index.json',capture/'source-index.json')
    registry=json.loads((OWNER/'registry.json').read_text());cases=registry['cases'];counts=collections.Counter(c['mode']for c in cases)
    assert len(cases)==registry['case_count']==70 and len({c['id']for c in cases})==70 and counts=={'scalar':44,'dual':26}
    lines=[]
    for c in cases:
        lines.append('CASE %s %s %d %d'%(c['id'],c['mode'],len(c['terms']),int(c['null_input'])))
        for term in c['terms']:
            assert len(term['atoms'])==4 and all(len(pair)==2 for pair in term['atoms'])
            lines.append('TERM %d %d %d %s'%(term['sign'],term['shift'],term['arity'],' '.join(word for pair in term['atoms']for word in pair)))
    assert '\n'.join(lines)+'\n'==(OWNER/'cases.txt').read_text()
    by_id={c['id']:c for c in cases}
    assert len(by_id['dual_max_generated_128']['terms'])==32 and all(t['arity']==4 for t in by_id['dual_max_generated_128']['terms'])
    for name in ['fraction_oracle.py','run_gate.py']:
        node=ast.parse((OWNER/name).read_text());assert not any(isinstance(n,ast.Import)and any(a.name.startswith(('numpy','scipy','mpmath'))for a in n.names)for n in ast.walk(node))
    recipe=json.loads((OWNER/'recipe.json').read_text())
    assert recipe['compiler']['path'].endswith('/clang') and '--driver-mode=g++'not in recipe['release_flags']+recipe['debug_flags']
    assert not any(x in recipe['release_flags']+recipe['debug_flags']for x in ['-lc++','-lstdc++'])
    assert recipe['fixed_cases']==70 and recipe['scalar_cases']==44 and recipe['dual_cases']==26
    assert '-fno-fast-math'in recipe['release_flags']and '-fno-fast-math'in recipe['debug_flags']
    after={p:sha(p)for p in pins};assert after==pins
    write(P/'inputs-after.json',after)
    receipt={'passed':False,'passed_source_review':False,'arithmetic_source_math_review_passed':True,
        'reviewed_source_index_sha256':EXPECTED,'source_inputs_unchanged':True,'protected_pins':len(pins),
        'registry_cases':70,'scalar_cases':44,'dual_cases':26,'full_128_product_rule_registry_present':True,
        'registry_text_exact':True,'compiler_or_arithmetic_executed':False,'candidate_imported':False,
        'scope':'source/math/admission review only; standalone submitted-atom primitive, no RWM repair or unit acceptance',
        'blocking_finding':'Compiler invocation uses literal clang C-driver basename to link a C++ standard-library probe; no C++ driver mode or standard-library linker flag. Root confirmed. Source001 remains unexecuted, not a compile failure.',
        'next_stage':'Fresh source002 literal clang++ invocation plus resolved compiler binary/hash guard, exact arithmetic/probe/oracle/registry equality, separate review before compile.',
        'seconds':time.monotonic()-t}
    write(P/'receipt.json',receipt);print(json.dumps(receipt,sort_keys=True))

if __name__=='__main__':main()
