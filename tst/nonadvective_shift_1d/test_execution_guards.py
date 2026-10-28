"""Negative integrity and capacity checks using temporary synthetic records."""
import argparse,json,tempfile
from pathlib import Path
from campaign import main_reservations,digest
from run_segment import verify
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();checks=[]
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);case=root/'case';case.mkdir();payload=case/'profiles.npz';payload.write_bytes(b'clearly synthetic verification fixture')
 m=dict(status='completed',checkpoint_hashes={'profiles.npz':digest(payload)},frozen_hashes={'source':'synthetic'})
 cp={'frozen_hashes':m['frozen_hashes']};(case/'manifest.json').write_text(json.dumps(m));assert verify(case,cp)==m;checks.append('completed integrity fixture accepted')
 for status in ['failed','bounded_stop']:
  bad=dict(m,status=status);(case/'manifest.json').write_text(json.dumps(bad))
  try:verify(case,cp)
  except ValueError:checks.append(status+' parent rejected')
  else:raise AssertionError('Noncompleted parent accepted')
 (case/'manifest.json').write_text(json.dumps(m));payload.write_bytes(b'tampered')
 try:verify(case,cp)
 except ValueError:checks.append('checkpoint tampering rejected')
 else:raise AssertionError('Tampering accepted')
 payload.write_bytes(b'clearly synthetic verification fixture')
 try:verify(case,{'frozen_hashes':{'source':'changed'}})
 except ValueError:checks.append('source binding mismatch rejected')
 else:raise AssertionError('Source mismatch accepted')
 records=['gamma4-duration-series-v1/n513/series.json','gamma4-duration-series-v1/n1025/series.json','gamma4-time-100M-queue-v1/queue.json']
 def write(index,status):
  p=root/records[index];p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(dict(status=status,pid=99999999,command=['nonexistent-synthetic-process'])))
 for i in range(3):write(i,'running')
 write(2,'waiting');assert len(main_reservations(root))==2;checks.append('waiting queue shares spatial gate reservation')
 write(0,'completed');write(2,'running_series');r=main_reservations(root);assert len(r)==2 and not any(x['command_verified'] for x in r);checks.append('missing handles do not release unresolved running reservations')
 write(2,'completed');assert len(main_reservations(root))==1;checks.append('verified terminal coordinator state releases one reserved slot')
a.output.write_text(json.dumps(dict(passed=len(checks),checks=checks,scope='Synthetic negative integrity/capacity tests only; no scientific data or trajectory was fabricated or modified.'),indent=2)+'\n');print(checks)
