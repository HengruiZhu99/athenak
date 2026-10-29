#!/usr/bin/env python3
"""Correlate verified rank costs with direct radial row ownership; no job launch."""
from pathlib import Path
import hashlib,json
root=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').is_file())
raw=root/'docs/hyperbolic-shift-3d/reproduction/aurora/preflight-measured-partition-v1'
verification=json.loads((raw/'verification.json').read_text());assert verification['passed']
records=[]
for name in ['puncture-control','puncture-fast']:
    p=raw/'output'/(name+'.report.json');digest=hashlib.sha256(p.read_bytes()).hexdigest()
    assert any(a['remote_path'].endswith('/'+p.name) and a['sha256']==digest for a in verification['artifacts'])
    r=json.loads(p.read_text());n=r['n'];cells=r['cells'];ranks=r['mpi_size']
    assert (n,cells,ranks,r['mesh_blocks'],r['cartesian_core_subdivisions'],r['scri_radial_bulk_rows'])==(16,159200,12,26,2,3)
    assert r['mesh_topology']=='cartesian_2x2x2_shells3' and r['scri_radial_closure']=='donor_seven_last3'
    # CompositeMesh appends 8 core blocks then 6 faces for each of 3 shells.
    # CompositeDifference appends all nominal block cells (i fastest), then
    # overlap cells. ScriRadialDifference selects i=n-3..n-1 of outer blocks.
    cuts=r.get('ownership_cuts',[cells*i//ranks for i in range(ranks+1)])
    rows=[]
    for rank,(a,b) in enumerate(zip(cuts,cuts[1:])):
        radial=sum(p%n>=n-3 for p in range(max(a,20*n**3),min(b,26*n**3)))
        rows.append(dict(rank=rank,owned_cells=b-a,direct_radial_rows=radial,rhs_seconds=r['phase_seconds_by_rank'][rank][0]))
    assert sum(x['direct_radial_rows'] for x in rows)==6*n*n*3
    records.append(dict(name=name,report_sha256=digest,rows=rows))
out=dict(scope='Derived ownership counts for the verified V21/V22 fixed mesh only. Cost correlation does not establish GPU occupancy, register spills, or compiler behavior.',
         ordering_sources=['src/mesh/cubed_sphere/composite.hpp','src/mesh/cubed_sphere/composite_difference.hpp','src/mesh/cubed_sphere/scri_radial_difference.hpp'],
         source_reference='85b8ba91a45dca07496d941e765ea6929006c352',records=records)
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
