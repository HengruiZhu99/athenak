"""Saved capsule hashes and declared48point gate receipts; no source call."""
from pathlib import Path
import argparse,hashlib,json,math
def main():
 p=argparse.ArgumentParser();p.add_argument('capsule');p.add_argument('--metadata-only',action='store_true');a=p.parse_args();P=Path(a.capsule);idx=json.loads((P/'index.json').read_text());count=0;skipped=[]
 for e in idx['files']:
  f=P/e['path']
  if not f.exists()and a.metadata_only and e['role']=='large_payload':skipped.append(e['path']);continue
  b=f.read_bytes();assert len(b)==e['bytes']and hashlib.sha256(b).hexdigest()==e['sha256'];count+=1
 r=json.loads((P/idx['accepted_attempt']/'receipt.json').read_text());assert r['passed']and r['release_debug_equal']and r['source_before']==r['source_after'];assert len(r['source_before'])==381
 for mode in ['release','debug']:
  report=json.loads((P/idx['accepted_attempt']/('analysis-'+mode+'/receipt.json')).read_text());assert report['passed']and report['cases']==48
  for key,t in report['thresholds'].items():assert math.isfinite(report['maxima'][key])and report['maxima'][key]<=t,key
 assert (P/idx['accepted_attempt']/'run-release.stdout').read_bytes()==(P/idx['accepted_attempt']/'run-debug.stdout').read_bytes()
 assert all(q['exit_code']==0 and q['stderr_sha256']==hashlib.sha256(b'').hexdigest()for q in r['commands'])
 print(json.dumps({'passed':True,'checked_files':count,'skipped_large_payloads':skipped,'scope':'saved hashes/thresholds only; no kernel query/compile/evolution'},indent=2))
if __name__=='__main__':main()
