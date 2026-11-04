"""Focused integrity checks for the new traversal; no scheduler or launcher test."""
from pathlib import Path
import tempfile,os,json,importlib.util
from unittest.mock import patch
import storage_accounting as new
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_storage',HERE/'launcher-v6/storage.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
with tempfile.TemporaryDirectory() as t:
 p=Path(t);a=p/'a';b=a/'b';outside=p/'outside';b.mkdir(parents=True);outside.mkdir();(a/'data').write_bytes(b'x'*8193);os.link(a/'data',b/'hardlink');(b/'small').write_bytes(b'abc');(outside/'external').write_bytes(b'y'*5017);(a/'external-link').symlink_to(outside,target_is_directory=True);(b/'cycle').symlink_to(a,target_is_directory=True);(b/'dangling').symlink_to(p/'missing');(p/'root-link').symlink_to(a,target_is_directory=True)
 def independent(paths):
  seen=set();total=0
  for path in paths:
   s=path.lstat();key=(s.st_dev,s.st_ino)
   if key not in seen:total+=max(s.st_size,s.st_blocks*512);seen.add(key)
  return total
 ordinary=[a,b,a/'data',b/'small',a/'external-link',b/'cycle',b/'dangling']
 cases=[([a,b,a],independent(ordinary)),([b,a],independent(ordinary)),([p/'root-link'],independent([p/'root-link']+[x for x in ordinary if x!=a])),([a,a/'external-link'],independent(ordinary+[outside/'external'])),([a,outside],independent(ordinary+[outside,outside/'external']))]
 results=[]
 for roots,expected in cases:
  old.PRIOR=roots[:-1];o=old.capacity(roots[-1],planned=111,reserve=222);n=new.accounted_bytes(roots);assert o['bytes']==n['bytes']==expected,(roots,o,n,expected);results.append(dict(roots=[str(x.relative_to(p)) for x in roots],bytes=expected,scan=n))
 try:new.accounted_bytes([a,p/'missing'])
 except RuntimeError:pass
 else:raise AssertionError('missing root silently accepted')
 orig=os.scandir
 def denied(path):
  if Path(path)==b:raise PermissionError('simulated unreadable accounted directory')
  return orig(path)
 with patch.object(os,'scandir',side_effect=denied):
  try:new.accounted_bytes([a])
  except PermissionError:pass
  else:raise AssertionError('unreadable directory silently accepted')
 new.PRIOR=[a];new.LIMIT=independent(ordinary)+333
 assert new.capacity(a,111,222)['headroom_bytes']==0
 try:new.capacity(a,112,222)
 except RuntimeError:pass
 else:raise AssertionError('over-cap accepted')
 print(json.dumps(dict(passed=True,cases=results,missing_root_rejected=True,unreadable_directory_rejected=True,exact_cap_and_over_cap_checked=True),indent=2))
