from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
O=P.parent/'continuum/native-angular-pulse-flat-derivatives-compact-root-launch-held-20261009'
I=P.parent/'continuum/native-angular-pulse-flat-derivatives-compact-root-independent-source-review-20261009'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P/'source-review001.json')=='edbf8dc769b678c5b6c08f7e55846dcbafb4bb752e926974cd309444722a7da0'
assert sha(I/'receipt.json')=='cd7ef0fbde22a965f70fa9aca1c089346db79cf19f901b6226f9e9aa1fa0b092'
assert sha(I/'index.json')=='12581fa5247efb7e7e30086decb9924c33319e31517f9849ce9b98c5a0ecfc79'
review=json.loads((I/'receipt.json').read_text())
assert review['disposition']=='PASS_SOURCE_ONLY_NO_EXECUTION' and review['blocking_corrections']==[]
auth=json.loads((O/'authorization-schema.json').read_text())
assert auth['candidate_source_index']['sha256']=='4f5a4fbbfa6c6e4af867963189f4d3a5549116e047849bf9fac8d3c96b6c9607'
for path,digest in {**auth['outer_source_pins'],**auth['source_pins']}.items(): assert sha(path)==digest,path
for key in ('fresh_invocation_path','fresh_output_path'): assert not Path(auth[key]).exists(),key
auth.update(kind='Root one-shot release after independent and root source review',one_shot_outer_compact_comparison_launch_admitted=True,compact_root_comparison_execution_admitted=True,authorization_path=str(P/'authorization.json'),root_review_sha256=sha(P/'source-review001.json'),independent_review_sha256=sha(I/'receipt.json'),full_derivative_gate_admitted=False)
with (P/'authorization.json').open('x') as f: f.write(json.dumps(auth,indent=2,allow_nan=False)+'\n')
release={'passed':True,'authorization_sha256':sha(P/'authorization.json'),'independent_review_sha256':sha(I/'receipt.json'),'root_review_sha256':sha(P/'source-review001.json'),'scope':'Sole480-ray old-output comparison only; full gate/inverse/evolution held'}
with (P/'release.json').open('x') as f: f.write(json.dumps(release,indent=2)+'\n')
print(json.dumps(release))
