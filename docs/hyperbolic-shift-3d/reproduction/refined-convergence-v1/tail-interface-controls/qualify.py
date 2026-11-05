"""Run existing frozen host configuration utility; no rebuild or GPU check."""
from pathlib import Path
import json,hashlib,subprocess,sys
root=Path(sys.argv[1]).resolve();package=Path(__file__).resolve().parent
launcher=root/'launcher-boosted-transition-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,record in json.loads((package/'package-records.json').read_text()).items():
    assert sha(package/name)==record['sha256'] and (package/name).stat().st_size==record['bytes']
for name,record in json.loads((launcher/'manifest.json').read_text()).items():
    assert sha(launcher/name)==record['sha256']
calibration=json.loads((launcher/'known-aurora-calibration.json').read_text())
for name in ['radial-graded-tail','angular-seam-inward']:
    local=json.loads((package/(name+'-local.json')).read_text())
    expected=local['identity']
    for change in calibration['changes']:
        assert expected.count(change['local'])==1
        expected=expected.replace(change['local'],change['aurora'])
    cfg=package/(name+'.cfg');assert sha(cfg)==local['config_sha256']
    target=package/(name+'-qualified.json');assert not target.exists()
    text=subprocess.check_output([str(launcher/'configuration-identity'),str(cfg)],text=True)
    (package/(name+'-identity.jsonl')).write_text(text)
    rows=[json.loads(x) for x in text.splitlines()]
    assert [x['n'] for x in rows]==[12,14,16]
    ids={x['identity'] for x in rows};assert len(ids)==1
    identity=ids.pop();assert identity in (local['identity'],expected)
    assert all(x['inner_gamma']==x['k']*x['k'] for x in rows)
    record=dict(local,identity=identity,configuration_utility_sha256=sha(launcher/'configuration-identity'),
                qualification='Actual frozen production compiler host identity at N12/14/16; N12 used. No GPU check, evolution, rebuild or freeze.')
    target.write_text(json.dumps(record,indent=2)+'\n')
    print(name,identity)
