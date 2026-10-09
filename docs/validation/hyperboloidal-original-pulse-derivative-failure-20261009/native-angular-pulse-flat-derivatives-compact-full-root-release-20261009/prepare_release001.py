from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
O=P.parent/'continuum/native-angular-pulse-flat-derivatives-compact-full-launch-held-20261009'
I=P.parent/'boundary/reference-wave-map-compact-full-independent-source-review-20261009/combined-review-receipt.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P/'source-review001.json')=='58a9390c0cf0a93059c4ad2a589782186973ab32eb8eb8e391ed0416e636c449'
assert sha(I)=='2d6ac9151be98a5194b142a31aa47416d09a998f978a957603772a4de7c56197'
r=json.loads(I.read_text());assert r['passed_independent_full_candidate_and_outer_source_review'] and r['all_pins_unchanged'] and r['blocking_corrections']==[]
auth=json.loads((O/'authorization-schema.json').read_text())
for path,digest in {**auth['outer_source_pins'],**auth['source_pins']}.items():assert sha(path)==digest,path
for key in ('fresh_invocation_path','fresh_output_path'):assert not Path(auth[key]).exists()
auth.update(kind='Root one-shot full scalar derivative gate release',authorization_path=str(P/'authorization.json'),one_shot_outer_derivative_launch_admitted=True,analytic_scalar_derivative_execution_admitted=True,root_review_sha256=sha(P/'source-review001.json'),independent_review_sha256=sha(I))
with (P/'authorization.json').open('x') as f:f.write(json.dumps(auth,indent=2,allow_nan=False)+'\n')
release={'passed':True,'auth_sha256':sha(P/'authorization.json'),'root_review_sha256':sha(P/'source-review001.json'),'independent_review_sha256':sha(I),'scope':'Full fixed scalar value/gradient/Hessian gate only; inverse/native/stability/BH held.'}
with (P/'release.json').open('x') as f:f.write(json.dumps(release,indent=2)+'\n')
print(json.dumps(release))
