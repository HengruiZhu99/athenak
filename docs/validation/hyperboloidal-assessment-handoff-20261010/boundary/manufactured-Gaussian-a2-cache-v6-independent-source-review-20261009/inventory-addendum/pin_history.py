import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'manufactured-Gaussian-a2-cache-v6-instrumented-held-20261009'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def pin(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 if (HERE/'index.json').exists():raise RuntimeError('one-shot addendum')
 if sha(OWNER/'source-index.json')!='cafc4df240ba9988919f2f4d97522bda2f71a19f1899a625d2be98afa2144397':raise RuntimeError('original index pin')
 original=json.loads((OWNER/'source-index.json').read_text())
 for r in original['files']:
  if sha(Path(r['path']))!=r['sha256']:raise RuntimeError('candidate pin')
 files=[OWNER/'source-index.json']+sorted(OWNER.glob('history/**/source-index.json'))
 if len(files)!=4:raise RuntimeError('fixed missing historical count')
 save(HERE/'additional-protected-history.json',{'files':[pin(p) for p in files],
  'all_active_candidate_sources_previously_indexed':True,'historical_additions_only':3,'candidate_source_modified':False})
 save(HERE/'receipt.json',{'passed_metadata_inventory_audit':True,'original_index_unchanged':True,
  'original_indexed_files':67,'additional_historical_files':3,'candidate_execution':False})
 own=sorted(p for p in HERE.iterdir() if p.is_file() and p!=HERE/'index.json')
 save(HERE/'index.json',{'status':'immutable additive history coverage','files':[pin(p) for p in own]})
 print(json.dumps({'index':sha(HERE/'index.json'),'history':sha(HERE/'additional-protected-history.json')}))
if __name__=='__main__':main()
