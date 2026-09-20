"""Private validation-only repartition of existing eight-rank saved data.

No numerical evolution or restart is performed. Source checkpoints are read only;
output fixture is marked ineligible for AthenaK and must never be evolved.
"""
import argparse,hashlib,json,shutil,struct
from pathlib import Path
import numpy as np
from check_smr_trumpet_checkpoint import read_header,validate,equal_partition,check_supported
from check_run import check
p=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--fixture',type=Path,required=True);a=ap.parse_args()
a.fixture.mkdir(exist_ok=False);(a.fixture/'VALIDATION_ONLY_DO_NOT_EVOLVE.txt').write_text('Synthetic repartition for parser tests only. Not a24-rank evolution or authorized restart.\n')
files=sorted(a.source.glob('rst/rank_*/ks_background.00001.rst'));assert len(files)==8
h=read_header(files[0]);assert h['total']==232 and h['cycle']==3
owners,counts=equal_partition(232,24);ng,n,ns,lo,hi,shape,offset=check_supported(h)
source_hashes={};readers=[]
for f in files:
 hr=read_header(f);assert hr['header']==h['header'];s=f.open('rb');s.seek(h['payload_start']);readers.append(s);source_hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
gid=0
for rank,count in enumerate(counts):
 dest=a.fixture/'rst'/f'rank_{rank:08d}'/files[0].name;dest.parent.mkdir(parents=True)
 with dest.open('wb')as out:
  out.write(h['header'])
  for _ in range(count):
   data=readers[gid//29].read(h['stride']);assert len(data)==h['stride'];out.write(data);gid+=1
for s in readers:s.close()
for f in a.source.glob('*.hst'):shutil.copy2(f,a.fixture/f.name)
shutil.copy2(a.source/'run.log',a.fixture/'run.log');(a.fixture/'exit_code.txt').write_text('0\n')
passed=check(a.fixture,True,24);assert passed['passed']and passed['residual_exactly_zero']and sum(passed['blocks_per_rank'])==232
# A single finest-level metric ghost is corrupted only in the private fixture.
block=next(b for b in json.loads((p/'mesh-audit.json').read_text())['cases']['mesh16_r1']['blocks_geometry']if b['dx_M']==.125 and b['min'][0]==-4)
gid=block['gid'];rank=int(owners[gid]);first=sum(counts[:rank]);local=gid-first;i,j,k=ng-1,ng+n[1]//2,ng+n[2]//2
index=((1*shape[0]+int(k))*shape[1]+int(j))*shape[2]+int(i);address=h['payload_start']+local*h['stride']+offset+8*index
file=a.fixture/'rst'/f'rank_{rank:08d}'/files[0].name
with file.open('r+b')as s:s.seek(address);saved=s.read(8);s.seek(address);s.write(struct.pack('<d',-2))
try:
 bad=validate(a.fixture,24,3,False,p/'mesh-audit.json');assert not bad['passed']and bad['invalid_metric_cells_including_ghosts']==1
finally:
 with file.open('r+b')as s:s.seek(address);s.write(saved)
last=a.fixture/'rst/rank_00000023'/files[0].name;held=last.with_suffix('.held');last.rename(held)
try:
 try:validate(a.fixture,24,3,True,p/'mesh-audit.json')
 except ValueError as exc:missing=str(exc)
 else:raise AssertionError('Missing rank accepted')
finally:held.rename(last)
# Header mismatch must also fail before a potentially stale cohort is accepted.
headerfile=a.fixture/'rst/rank_00000001'/files[0].name;cycleoff=h['header'].find(b'<par_end>\n')+len(b'<par_end>\n')+8+72+76+76+16
with headerfile.open('r+b')as s:s.seek(cycleoff);saved=s.read(4);s.seek(cycleoff);s.write(struct.pack('<i',99))
try:
 try:validate(a.fixture,24,3,True,p/'mesh-audit.json')
 except ValueError as exc:mismatch=str(exc)
 else:raise AssertionError('Mixed checkpoint cycles accepted')
finally:
 with headerfile.open('r+b')as s:s.seek(cycleoff);s.write(saved)
unchanged=all(hashlib.sha256(f.read_bytes()).hexdigest()==source_hashes[str(f)]for f in files);assert unchanged
result={'passed':True,'scope':'Parser/ownership/guard test using synthetic repartition of existing eight-rank raw checkpoint. Not a24-rank evolution. Fixture is never passed to AthenaK.','blocks':232,'ranks':24,'counts':counts,'full_finite_exact_zero_and_raw_ghost_SPD':True,'classification':passed['classification'],'single_fine_ghost_rejected':bad['invalid_metric_samples'][0],'missing_rank_rejected':missing,'mixed_cycle_header_rejected':mismatch,'original_eight_rank_files_unchanged':unchanged,'fixture':str(a.fixture)};(p/'validator24-regression.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
