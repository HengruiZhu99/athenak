#!/usr/bin/env python3
"""Read-only structural audit of AMR1 layouts; no evolution or timing claims."""
import argparse,collections,hashlib,json,pathlib,struct
p=argparse.ArgumentParser();p.add_argument('--runs',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
if a.output.exists():raise ValueError('Use a fresh output file')
def string(f):
    b=bytearray()
    while True:
        c=f.read(1)
        if not c:raise ValueError('truncated string')
        if c==b'\0':return b.decode()
        b.extend(c)
def audit(path):
    with path.open('rb') as f:
        science=string(f);assert science.startswith('HYP3D_AMR1;')
        names=[string(f) for _ in range(31)]
        fmt='<iiiQQddd3d8d8d';meta=struct.unpack(fmt,f.read(struct.calcsize(fmt)));n=meta[0]
        count=struct.unpack('<Q',f.read(8))[0];leaves=[struct.unpack('<4i',f.read(16)) for _ in range(count)]
        f.read(4*count);values=struct.unpack('<Q',f.read(8))[0]
    # CompositeDifference appends every nominal block, then evolved overlaps.
    kinds=['cartesian']*count+['shell_'+str(layer) for layer in range(6) for face in range(6)]
    sizes=[n**3]*len(kinds);spans=[(m,n**3) for m in range(len(kinds))]
    for m,(x,y,z,level) in enumerate(leaves):
        ext=1
        for loc in (x,y,z):ext*=n+3*(loc==0)+3*(loc==2**level-1)
        extra=ext-n**3
        if extra:spans.append((m,extra));sizes[m]+=extra
    for m in range(count,count+6):
        extra=(n+3)*(n+6)**2-n**3;spans.append((m,extra));sizes[m]+=extra
    total=sum(size for _,size in spans);assert total*31==values,(total,values)
    cases=[]
    for ranks in (1,2,4,8,12,24):
        cuts=[total*r//ranks for r in range(ranks+1)];owners=[]
        for rank in range(ranks):
            counts=collections.Counter();blocks=set();at=0
            for block,size in spans:
                overlap=max(0,min(at+size,cuts[rank+1])-max(at,cuts[rank]));at+=size
                if overlap:counts[kinds[block]]+=overlap;blocks.add(block)
            capacity=max(32,96//ranks+8)
            owners.append(dict(rank=rank,active=cuts[rank+1]-cuts[rank],by_region=dict(counts),selected_blocks=len(blocks),block_index_span=max(blocks)-min(blocks)+1,selected_whole_block_active=sum(sizes[b] for b in blocks),native_capacity=capacity,native_state_compensation_bytes=2*capacity*31*((n+8)**3+(n//2+8)**3)*8))
        cases.append(dict(ranks=ranks,default_constructor_can_start=ranks<=meta[2]**3,owners=owners))
    return dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),n=n,root_subdivisions=meta[2],leaves=count,active=total,blocks=len(kinds),two_full_padded_workspaces_bytes_per_rank=2*len(kinds)*(n+14)**3*31*8,state_and_base_bytes_per_rank=2*total*31*8,partitions=cases)
paths=['restart-v1/wave-full.checkpoint','controls-v1/boost-fixed.checkpoint','restart-v1/wave-save.checkpoint','controls-v1/boost-uniform.checkpoint']
result=dict(scope='Structural counts only. Hypothetical cuts at unsupported rank counts are not successful MPI runs. Whole-block active dependencies exclude additional halo/normal/scri donors; workspace bytes exclude geometry/plans and other buffers.',source_commit='c16a7c212acb00e9f963cb9a9f42d5d886777a7e',layouts=[audit(a.runs/path) for path in paths])
a.output.write_text(json.dumps(result,indent=2)+'\n')
for x in result['layouts']:
    q=next(p for p in x['partitions'] if p['ranks']==4)
    print('leaves',x['leaves'],'active',x['active'],'two padded MiB/rank',x['two_full_padded_workspaces_bytes_per_rank']/2**20,'4-rank owned/whole-block:',[(r['active'],r['selected_whole_block_active'],r['selected_blocks'],r['block_index_span']) for r in q['owners']])
