"""Execute input-parse/mesh-only audits; never initialize physics or evolve."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess
import numpy as np
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--binary',type=Path,required=True);ap.add_argument('--mpi-ranks',type=int,default=4);a=ap.parse_args();exe=a.binary.resolve()
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
records={'scope':'Input parse and mesh construction only. No physics constructor, initial-condition check, evolution, restart or job submission.','binary':str(exe),'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'cases':{}}
parse=P/'parse';parse.mkdir(exist_ok=True)
for inp in sorted((P/'inputs').glob('*.athinput')):
 r=subprocess.run([str(exe),'-i',str(inp),'-n'],cwd=parse,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (parse/(inp.stem+'.log')).write_text(r.stdout);assert r.returncode==0 and 'FATAL ERROR'not in r.stdout,(inp,r.stdout)
for name,stem,bs in [('mesh16','pulse_sponge',16),('mesh32','mesh32_pulse_sponge',32),('mesh32_inset','mesh32_inset_pulse_sponge',32)]:
 for ranks in [1,a.mpi_ranks]:
  run=P/f'{name}-r{ranks}';run.mkdir(exist_ok=True)
  cmd=(['mpiexec','-n',str(ranks)]if ranks>1 else[])+[str(exe),'-i',str(P/'inputs'/f'{stem}.athinput'),'-m']
  r=subprocess.run(cmd,cwd=run,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  (run/'mesh.log').write_text(r.stdout);assert r.returncode==0 and 'FATAL ERROR'not in r.stdout,(cmd,r.stdout)
  raw=(run/'mesh_structure.dat').read_text();blocks=[]
  for block in raw.split('#MeshBlock ')[1:]:
   lines=block.splitlines();gid,rank=re.match(r'(\d+) on rank=(\d+)',lines[0]).groups();level=int(re.search(r'Logical level (\d+)',lines[1])[1]);loc=list(map(int,re.search(r'location = \((.*?)\)',lines[1])[1].split()));xyz=np.array([[float(q)for q in l.split()]for l in lines[2:]if l.strip()]);lo=xyz.min(0);hi=xyz.max(0);dx=(hi-lo)/bs;assert np.all(dx==dx[0]);blocks.append({'gid':int(gid),'rank':int(rank),'logical_level':level,'logical_location':loc,'min':lo.tolist(),'max':hi.tolist(),'dx_M':float(dx[0])})
  assert len({b['gid']for b in blocks})==len(blocks)
  # Full-volume/no-interior-overlap audit of the exact output boxes.
  total_volume=sum(np.prod(np.array(b['max'])-b['min'])for b in blocks);assert total_volume==64**3
  for i,b in enumerate(blocks):
   for c in blocks[i+1:]:assert not np.all(np.minimum(b['max'],c['max'])>np.maximum(b['min'],c['min'])),(b,c)
  fine=[b for b in blocks if b['dx_M']==.125];fine_lo=np.min([b['min']for b in fine],axis=0);fine_hi=np.max([b['max']for b in fine],axis=0)
  fine_volume=sum(np.prod(np.array(b['max'])-b['min'])for b in fine);assert fine_volume==np.prod(fine_hi-fine_lo);assert np.all(fine_lo<=-4)and np.all(fine_hi>=4)
  assert not any(all(np.any(b['min'][axis]+(.5+np.arange(-4,bs+4))*b['dx_M']==0)for axis in range(3))for b in blocks) # no active or ghost cell center at puncture
  occupancy=[sum(b['rank']==rank for b in blocks)for rank in range(ranks)]
  n=bs+8;faces=3*(n+1)*n*n;rst_bytes=len(blocks)*8*(30*n**3+faces)
  per_dx={str(dx):sum(b['dx_M']==dx for b in blocks)for dx in sorted({b['dx_M']for b in blocks},reverse=True)}
  rec={'command':cmd,'returncode':r.returncode,'mesh_sha256':hashlib.sha256(raw.encode()).hexdigest(),'blocks':len(blocks),'blocks_by_dx':per_dx,'ranks':ranks,'blocks_per_rank':occupancy,'active_cells':len(blocks)*bs**3,'fine_ghost_cells':len(blocks)*n**3,'ghost_to_active_ratio':n**3/bs**3,'finest_cube_min_M':fine_lo.tolist(),'finest_cube_max_M':fine_hi.tolist(),'protected_sphere_radius_M':4,'protected_sphere_entirely_at_dx_0125':True,'no_active_or_ghost_center_at_puncture':True,'coordinate_horizon_radius_M':1,'diameter_over_dx':16,'fine_cube_coverage_no_gaps':True,'all_boxes_cover_domain_without_overlap':True,'finest_cell_center_radius_min_M':np.sqrt(3)*.125/2,'snapshot_payload_bytes_all_ranks_estimate':rst_bytes,'blocks_geometry':blocks}
  records['cases'][f'{name}_r{ranks}']=rec
  print(name,ranks,len(blocks),per_dx,'fine cube',fine_lo,fine_hi,'cells',rec['active_cells'],'ghostcells',rec['fine_ghost_cells'])
# Insets must be assessed on geometry, not on text.
x=records['cases']['mesh32_r1'];y=records['cases']['mesh32_inset_r1'];records['inset_geometry_equal']=x['blocks_geometry']==y['blocks_geometry']
(P/'mesh-audit.json').write_text(json.dumps(records,indent=2)+'\n')
