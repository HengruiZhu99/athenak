#!/usr/bin/env python3
"""Read-only validator for centered static-SMR M=R0=1 trumpet vacuum restarts.

Separate from the uniform checker. Supported format: little-endian doubles,
5-variable MHD plus 25 residual Z4c fields, per-rank files, fixed static mesh,
constant equal block costs and contiguous gid partition. No repairs/floors.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def as_bool(params, key, default=None):
    raw = params.get(key, default)
    if isinstance(raw, bool):
        return raw
    require(raw is not None, f'Missing boolean {key}')
    require(str(raw).lower() in ('true', 'false', '1', '0'), f'Invalid boolean {key}')
    return str(raw).lower() in ('true', '1')


def read_header(path):
    with Path(path).open('rb') as stream:
        prefix = stream.read(262144)
        marker = b'<par_end>\n'
        stop = prefix.find(marker)
        require(stop >= 0, 'Missing/oversized parameter header')
        end = stop + len(marker)
        params, section = {}, None
        for line in prefix[:stop].decode().splitlines():
            line = line.split('#', 1)[0].strip()
            if line.startswith('<'):
                section = line[1:-1]
                require(section not in params, 'Duplicate section')
                params[section] = {}
            elif '=' in line:
                k, v = line.split('=', 1)
                require(section is not None and k.strip() not in params[section], 'Duplicate/misplaced parameter')
                params[section][k.strip()] = v.strip()
        stream.seek(end)
        total, root = struct.unpack('<ii', stream.read(8))
        require(0 < total < 1000000 and 0 <= root < 31, 'Invalid mesh count/root level')
        region = struct.unpack('<9d', stream.read(72))
        root_indices = struct.unpack('<19i', stream.read(76))
        block_indices = struct.unpack('<19i', stream.read(76))
        time, dt, cycle = struct.unpack('<ddi', stream.read(20))
        locations = np.frombuffer(stream.read(16*total), dtype='<i4').reshape(total,4).copy()
        costs = np.frombuffer(stream.read(4*total), dtype='<f4').copy()
        output_times = struct.unpack('<2d', stream.read(16))
        stride, = struct.unpack('<Q', stream.read(8))
        payload_start = stream.tell()
        stream.seek(0)
        header = stream.read(payload_start)
    require(len(header) == payload_start, 'Truncated checkpoint header')
    return dict(path=str(path), params=params, total=total, root_level=root,
                region=region, root_indices=root_indices, indices=block_indices,
                time=time, checkpoint_dt=dt, cycle=cycle, locations=locations,
                costs=costs, output_times=output_times, stride=stride,
                payload_start=payload_start, header=header,
                header_sha256=hashlib.sha256(header).hexdigest())


def check_supported(h):
    p = h['params']
    require({'mesh','meshblock','mesh_refinement','mhd','z4c','coord','problem'} <= set(p), 'Missing required sections')
    require(not({'hydro','radiation','turbulence'} & set(p)), 'Unsupported additional checkpoint module')
    z, b, c, mesh = p['z4c'], p['problem'], p['coord'], p['mesh']
    require(p['mesh_refinement']['refinement'] == 'static', 'Only fixed static-SMR grids supported; no AMR/uniform fallback')
    require(b.get('pgen_name') == 'z4c_tov_ks' and b.get('bh_background') == 'schwarzschild_trumpet', 'Unsupported background')
    require(float(b['bh_mass']) == 1 and float(b['bh_spin']) == 0, 'Requires M=R0=1 spin0 trumpet')
    require(all(float(b[f'bh_center_x{i}']) == 0 for i in (1,2,3)), 'Requires centered background')
    require(as_bool(b,'pure_background') and as_bool(b,'zero_tmunu') and as_bool(p['mhd'],'zero_tmunu_feedback'), 'Only decoupled vacuum background controls supported')
    require(as_bool(b,'use_direct_z4c_background') and as_bool(z,'use_analytic_background'), 'Requires direct analytic residual background')
    require(not as_bool(c,'minkowski',False) and as_bool(c,'general_rel') and as_bool(c,'is_dynamical'), 'Unsupported coordinate reconstruction')
    require(float(c.get('a',0)) == 0 and not as_bool(c,'excise',False), 'Unsupported coordinate excision/spin')
    require(not as_bool(b,'force_minkowski_metric',False) and not as_bool(b,'excision_project_state'), 'Unsupported background projection')
    require(all(float(b[k]) == 0 for k in ('excision_freeze_radius','excision_ramp_radius','excision_damp_rate')), 'Inner treatment unsupported')
    require(float(z['chi_psi_power']) == -4, 'Requires chi=psi^-4')
    evolve = as_bool(z,'evolve_gauge_residual',True)
    require(as_bool(z,'evolve_lapse_residual',evolve) or as_bool(z,'preserve_lapse_residual',False), 'Stored lapse must contribute to full state')
    require(not any(k.startswith('co_') and k.endswith('_type') for k in z), 'Puncture tracker metadata unsupported')
    require(not any(k.startswith('dump_horizon_') and str(v).lower() in ('true','1') for k,v in z.items()), 'Horizon output metadata unsupported')
    require(int(p['mhd'].get('nscalars','0')) == 0, 'Only five MHD cell variables supported')
    require(sys.byteorder == 'little', 'Requires little-endian host')
    require(all(math.isfinite(x) for x in (*h['region'],h['time'],h['checkpoint_dt'])), 'Nonfinite header geometry/time')
    require(h['time'] >= 0 and h['cycle'] >= 0 and h['checkpoint_dt'] > 0, 'Invalid time/cycle/dt')
    ng,nx,ny,nz = h['indices'][:4]
    require(ng == 4 and min(nx,ny,nz) >= 8 and all(n%2 == 0 for n in (nx,ny,nz)), 'Requires even 3D blocks and four ghost layers')
    require(tuple(h['indices'][4:10]) == (ng,ng+nx-1,ng,ng+ny-1,ng,ng+nz-1), 'Inconsistent active block bounds')
    require(tuple(h['indices'][10:]) == (nx//2,ny//2,nz//2,ng,ng+nx//2-1,ng,ng+ny//2-1,ng,ng+nz//2-1), 'Inconsistent SMR coarse block bounds')
    require(tuple(h['root_indices'][10:]) == (0,)*9, 'Unexpected unused root coarse metadata')
    ns = np.array([int(mesh[f'nx{i}']) for i in (1,2,3)])
    n = np.array([nx,ny,nz]); lo = np.array([float(mesh[f'x{i}min']) for i in (1,2,3)]); hi = np.array([float(mesh[f'x{i}max']) for i in (1,2,3)])
    require(np.array_equal(ns, h['root_indices'][1:4]) and h['root_indices'][0] == ng, 'Root input/header mismatch')
    require(all(int(p['meshblock'][f'nx{i+1}']) == n[i] for i in range(3)), 'Block input/header mismatch')
    require(np.all(ns%n == 0) and np.all(hi>lo), 'Invalid root tiling')
    require(np.array_equal(lo,h['region'][:3]) and np.array_equal(hi,h['region'][3:6]), 'Region input/header mismatch')
    require(np.array_equal((hi-lo)/ns,h['region'][6:]), 'Root spacing mismatch')
    require(h['root_level'] == math.ceil(math.log2(int(max(ns//n)))), 'Unexpected root logical level')
    shape = (nz+2*ng,ny+2*ng,nx+2*ng); cells = math.prod(shape)
    faces = (shape[2]+1)*shape[1]*shape[0]+shape[2]*(shape[1]+1)*shape[0]+shape[2]*shape[1]*(shape[0]+1)
    offset = 8*(5*cells+faces)
    require(h['stride'] == offset+8*25*cells, 'Unsupported checkpoint payload stride/precision')
    require(np.isfinite(h['costs']).all() and np.all(h['costs'] == 1), 'Only static equal unit-cost partition supported')
    return ng,n,ns,lo,hi,shape,offset


def geometry(h, n, ns, lo, hi):
    blocks=[]; location_set=set()
    for gid,loc in enumerate(h['locations']):
        lx,ly,lz,level = map(int,loc); rel=level-h['root_level']
        require(0<=rel<=20, 'Unsupported/reflected refinement level')
        require(tuple(loc) not in location_set, 'Duplicate leaf logical location');location_set.add(tuple(loc))
        maxloc=(ns//n)*(2**rel)
        require(np.all(np.array([lx,ly,lz])>=0) and np.all(np.array([lx,ly,lz])<maxloc), 'Leaf logical location outside root domain')
        dx=(hi-lo)/ns/(2**rel); bmin=lo+np.array([lx,ly,lz])*n*dx; bmax=bmin+n*dx
        blocks.append(dict(gid=gid,logical_level=level,relative_level=rel,logical_location=[lx,ly,lz],min=bmin.tolist(),max=bmax.tolist(),dx_M=dx.tolist()))
    # Exact dyadic box checks: guard omitted parents/duplicate overlapping leaves.
    volume=sum(np.prod(np.array(b['max'])-b['min']) for b in blocks)
    require(abs(volume-np.prod(hi-lo)) <= 1e-12*np.prod(hi-lo), 'Leaves do not cover domain volume')
    for i,b in enumerate(blocks):
        for c in blocks[i+1:]:
            require(not np.all(np.minimum(b['max'],c['max'])>np.maximum(b['min'],c['min'])), 'Overlapping leaf interiors')
    return blocks


def equal_partition(total, ranks):
    require(0 < ranks <= total, 'Invalid MPI rank count')
    owners=np.empty(total,dtype=int);rank=ranks-1;left=np.float32(total);target=np.float32(left/ranks);cost=np.float32(0)
    for gid in range(total-1,-1,-1):
        cost=np.float32(cost+1);owners[gid]=rank
        if cost>=target and rank>0:
            rank-=1;left=np.float32(left-cost);cost=np.float32(0);target=np.float32(left/(rank+1))
    counts=[int(np.count_nonzero(owners==rank))for rank in range(ranks)]
    require(all(counts) and np.all(np.diff(owners)>=0), 'Invalid contiguous gid partition')
    return owners,counts


def validate(run, ranks, cycle=None, exact_zero=False, audit_path=None, audit_case='mesh16_r1'):
    run=Path(run);candidates=list((run/'rst/rank_00000000').glob('*.rst'));require(candidates,'No rank0 checkpoints')
    headers=[read_header(path)for path in candidates]
    if cycle is None:cycle=max(h['cycle']for h in headers)
    choices=[h for h in headers if h['cycle']==cycle];require(len(choices)==1,'Missing/ambiguous requested checkpoint cycle');h=choices[0]
    ng,n,ns,lo,hi,shape,offset=check_supported(h);blocks=geometry(h,n,ns,lo,hi);owners,counts=equal_partition(h['total'],ranks)
    files=[run/'rst'/f'rank_{rank:08d}'/Path(h['path']).name for rank in range(ranks)]
    require(set((run/'rst').glob('rank_*/'+Path(h['path']).name))==set(files),'Missing/extra rank-file cohort')
    audit_match=None
    if audit_path:
        audit=json.loads(Path(audit_path).read_text())['cases'][audit_case];ab={b['gid']:b for b in audit['blocks_geometry']}
        require(set(ab)==set(range(len(blocks))),'Audit gid set differs')
        for b in blocks:
            q=ab[b['gid']]
            require(b['logical_level']==q['logical_level'] and b['logical_location']==q['logical_location'] and b['min']==q['min'] and b['max']==q['max'] and all(dx==q['dx_M'] for dx in b['dx_M']),f'Mesh-audit geometry mismatch at gid{b["gid"]}')
        audit_match=True
    minima=dict.fromkeys(['alpha','chi','gxx','second_minor','determinant'],np.inf);bad=0;samples=[];file_records=[];nonzero=0;active_nonzero=0;field_max=np.zeros(25);field_max_active=np.zeros(25);payload_finite=True;gid_start=0
    for rank,path in enumerate(files):
        hr=read_header(path);require(hr['header']==h['header'],f'Rank{rank} cohort header differs')
        expected=h['payload_start']+counts[rank]*h['stride'];require(path.stat().st_size==expected,f'Rank{rank} payload count does not match deterministic contiguous gids')
        payload=np.memmap(path,dtype='<f8',mode='r',offset=h['payload_start'],shape=(counts[rank],h['stride']//8));finite=bool(np.isfinite(payload).all());payload_finite &= finite
        for local in range(counts[rank]):
            gid=gid_start+local;b=blocks[gid];require(int(owners[gid])==rank,'Ownership index mismatch')
            u=payload[local,offset//8:].reshape((25,*shape));active=u[:,ng:ng+int(n[2]),ng:ng+int(n[1]),ng:ng+int(n[0])]
            nonzero+=int(np.count_nonzero(u));active_nonzero+=int(np.count_nonzero(active));field_max=np.maximum(field_max,np.max(np.abs(u),axis=(1,2,3)));field_max_active=np.maximum(field_max_active,np.max(np.abs(active),axis=(1,2,3)))
            axes=[b['min'][a]+(np.arange(int(n[a])+2*ng)-ng+.5)*b['dx_M'][a]for a in range(3)]
            z,y,x=np.meshgrid(axes[2],axes[1],axes[0],indexing='ij');r=np.sqrt(x*x+y*y+z*z);require(np.all(r>0),'Active/ghost center lies at puncture')
            alpha=r/(r+1)+u[18];chi=(r/(r+1))**2+u[0];xx,xy,xz,yy,yz,zz=1+u[1],u[2],u[3],1+u[4],u[5],1+u[6];minor=xx*yy-xy*xy;det=xx*yy*zz+2*xy*xz*yz-xx*yz*yz-yy*xz*xz-zz*xy*xy
            values=dict(alpha=alpha,chi=chi,gxx=xx,second_minor=minor,determinant=det);invalid=np.zeros(shape,dtype=bool)
            for key,v in values.items():invalid |= ~np.isfinite(v)|(v<=0);minima[key]=min(minima[key],float(np.min(v)))
            bad+=int(invalid.sum())
            for k,j,i in np.argwhere(invalid)[:max(0,16-len(samples))]:
                ijk=(int(i),int(j),int(k));samples.append(dict(rank=rank,gid=gid,local_block=local,logical_level=b['logical_level'],relative_level=b['relative_level'],array_ijk=list(ijk),xyz_M=[float(axes[a][ijk[a]])for a in range(3)],ghost_depth=[max(ng-ijk[a],ijk[a]-(ng+int(n[a])-1),0)for a in range(3)],raw_full_values={key:float(v[k,j,i])for key,v in values.items()}))
        sha=hashlib.sha256()
        with path.open('rb')as stream:
            for data in iter(lambda:stream.read(8*1024*1024),b''):sha.update(data)
        file_records.append(dict(rank=rank,path=str(path),bytes=path.stat().st_size,sha256=sha.hexdigest(),gid_first=gid_start,gid_last=gid_start+counts[rank]-1,blocks=counts[rank],all_payload_finite=finite));gid_start+=counts[rank];del payload
    zero=(nonzero==0);passed=payload_finite and bad==0 and (zero or not exact_zero)
    return dict(passed=bool(passed),time_M=h['time'],cycle=h['cycle'],checkpoint_dt=h['checkpoint_dt'],root_logical_level=h['root_level'],relative_levels=sorted(set(b['relative_level']for b in blocks)),blocks=h['total'],ranks=ranks,blocks_per_rank=counts,matching_cohort_headers=True,header_sha256=h['header_sha256'],contiguous_gid_partition_counts_match=True,ownership_scope='Rank-file payload order is implicit in the format; verified file paths, equal costs, deterministic contiguous counts and common location table. Format has no embedded per-block gid or rank signature.',mesh_audit_geometry_matches=audit_match,all_payload_finite=payload_finite,invalid_metric_cells_including_ghosts=bad,invalid_metric_samples=samples,minimum_raw_full=minima,residual_exactly_zero=zero,residual_nonzero_count_including_ghosts=nonzero,residual_nonzero_count_active=active_nonzero,residual_field_max_including_ghosts=field_max.tolist(),residual_field_max_active=field_max_active.tolist(),files=file_records,scope='Saved static-SMR residual payload and raw alpha/chi/conformal metric Sylvester minors only; no floors, clipping, constraint or stability clearance.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);ap.add_argument('--ranks',type=int,required=True);ap.add_argument('--cycle',type=int);ap.add_argument('--exact-zero',action='store_true');ap.add_argument('--mesh-audit',type=Path);ap.add_argument('--audit-case',default='mesh16_r1');ap.add_argument('--output',type=Path);a=ap.parse_args()
    try:result=validate(a.run,a.ranks,a.cycle,a.exact_zero,a.mesh_audit,a.audit_case)
    except(Exception)as exc:result={'passed':False,'validation_error':str(exc),'error_type':type(exc).__name__}
    def safe_json(value):
        if isinstance(value,float) and not math.isfinite(value):return str(value)
        if isinstance(value,dict):return {key:safe_json(item)for key,item in value.items()}
        if isinstance(value,list):return [safe_json(item)for item in value]
        return value
    result=safe_json(result)
    output=a.output or a.run/'checkpoint-validity.json';output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['files','residual_field_max_including_ghosts','residual_field_max_active']},indent=2,allow_nan=False));raise SystemExit(0 if result['passed']else 1)
