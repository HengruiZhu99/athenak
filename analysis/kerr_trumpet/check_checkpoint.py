#!/usr/bin/env python3
"""Read-only checkpoint audit for centered M=R0=1, a=.9 Kerr trumpet controls.

Supports little-endian doubles, five MHD cell fields, magnetic face fields and
25 residual Z4c fields on uniform/static dyadic meshes with equal unit costs.
Only the binary-format reader and deterministic partition routine are imported
from the older campaign. No Schwarzschild background validator is called.
No checkpoint is repaired; this audit is not a resubmission controller.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tde_revalidation'))
from validate_checkpoint import read_header, equal_partition


def require(ok, message):
    if not ok:
        raise ValueError(message)


def boolean(p, key, default=None):
    value = str(p.get(key, default)).lower()
    require(value in ('true', 'false', '1', '0'), 'Invalid/missing boolean ' + key)
    return value in ('true', '1')


def determinant(g):
    xx, xy, xz, yy, yz, zz = g
    # Same scalar expression as adm::SpatialDet, independently evaluated in NumPy.
    return -(xz*xz)*yy + 2*xy*xz*yz - (yz*yz)*xx - (xy*xy)*zz + xx*yy*zz


def background(x, y, z, spin):
    """DBM Eq.12--18, R0=M=1, followed by the stored metric projection.

    NumPy and C++ scalar libm operations need not agree in their last bits.
    This is an independent positive-metric reconstruction, not a bitwise
    background equivalence test. No lapse/chi/metric floors are applied.
    """
    r2 = x*x + y*y + z*z
    require(np.all(r2 > 0), 'A saved active/ghost cell is at the puncture')
    r = np.sqrt(r2); R = r + 1
    n = [x/r, y/r, z/r]; w = [-n[1], n[0], np.zeros_like(r)]
    a2 = spin*spin; gap = math.sqrt(1-a2)
    sigma = R*R + a2*n[2]*n[2]
    big_x = (R*R+a2)*(R*R+a2) - a2*(x*x+y*y)
    alpha = r*np.sqrt(sigma/big_x)
    det_physical = sigma*big_x/(r2*r2*r2)
    chi = np.power(det_physical, -1.0/3.0)
    metric = []
    for i,j in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]:
        gij = (sigma*(i==j) + a2*(1+2*R/sigma)*w[i]*w[j]
               - spin*gap*(n[i]*w[j]+w[i]*n[j]))/r2
        metric.append(chi*gij)
    det = determinant(metric)
    require(np.all(np.isfinite(det)) and np.all(det > 0), 'Invalid analytic background metric')
    scale = np.cbrt(1/det)
    metric = [g*scale for g in metric]
    return r, alpha, chi, metric


def supported(h):
    p = h['params']
    require({'mesh','meshblock','mesh_refinement','mhd','z4c','coord','problem'} <= set(p), 'Missing sections')
    require(not ({'hydro','radiation','turbulence'} & set(p)), 'Unsupported extra checkpoint module')
    mesh, block, mr, mhd, z, c, b = [p[s] for s in ('mesh','meshblock','mesh_refinement','mhd','z4c','coord','problem')]
    require(mr.get('refinement') in ('none','static'), 'Only uniform/static controls supported')
    require(not boolean(mr,'prolong_primitives',False), 'Primitive prolongation unsupported')
    require(b.get('pgen_name') == 'z4c_tov_ks' and b.get('bh_background') == 'kerr_trumpet', 'Requires new Kerr trumpet background')
    spin = float(b['bh_spin'])
    require(float(b['bh_mass']) == 1 and spin == .9, 'Scope requires M=R0=1 and a=.9')
    require(all(float(b.get('bh_center_x'+str(i),'0')) == 0 for i in (1,2,3)), 'Requires centered background')
    require(boolean(b,'use_direct_z4c_background') and boolean(z,'use_analytic_background'), 'Requires direct residual background')
    require(not boolean(c,'minkowski',False) and not boolean(c,'excise',False) and float(c['a']) == spin, 'Coordinate flags/spin unsupported')
    require(not boolean(b,'force_minkowski_metric',False) and not boolean(b,'excision_project_state'), 'Background/core projection unsupported')
    require(all(float(b[k]) == 0 for k in ('excision_freeze_radius','excision_ramp_radius','excision_damp_rate')), 'Inner treatment unsupported')
    require(float(z['chi_psi_power']) == -4 and not boolean(z,'residual_hamiltonian_balance',False), 'Unsupported conformal representation/balancing')
    gauge = boolean(z,'evolve_gauge_residual',True)
    require(boolean(z,'evolve_lapse_residual',gauge) or boolean(z,'preserve_lapse_residual',False), 'Stored lapse must contribute to full state')
    require(not any(k.startswith('co_') and k.endswith('_type') for k in z), 'Tracker metadata unsupported')
    require(not any(k.startswith('dump_horizon_') and str(v).lower() in ('true','1') for k,v in z.items()), 'Horizon metadata unsupported')
    require(int(mhd.get('nscalars',0)) == 0 and sys.byteorder == 'little', 'Requires five MHD fields and little-endian doubles')
    require(all(math.isfinite(v) for v in (*h['region'],h['time'],h['checkpoint_dt'])), 'Nonfinite header geometry/time')
    require(h['time'] >= 0 and h['cycle'] >= 0 and h['checkpoint_dt'] > 0, 'Invalid time/cycle/dt')
    ng,nx,ny,nz = h['indices'][:4]; n = np.array([nx,ny,nz])
    require(ng == 4 and min(n) >= 8 and np.all(n%2 == 0), 'Requires even 3D blocks and four ghosts')
    require(tuple(h['indices'][4:10]) == (ng,ng+nx-1,ng,ng+ny-1,ng,ng+nz-1), 'Invalid active bounds')
    require(tuple(h['indices'][10:]) == (nx//2,ny//2,nz//2,ng,ng+nx//2-1,ng,ng+ny//2-1,ng,ng+nz//2-1), 'Invalid coarse block bounds')
    require(tuple(h['root_indices'][10:]) == (0,)*9, 'Unexpected root coarse metadata')
    ns = np.array([int(mesh['nx'+str(i)]) for i in (1,2,3)])
    lo = np.array([float(mesh['x'+str(i)+'min']) for i in (1,2,3)])
    hi = np.array([float(mesh['x'+str(i)+'max']) for i in (1,2,3)])
    require(h['root_indices'][0] == ng and np.array_equal(ns,h['root_indices'][1:4]), 'Root input/header mismatch')
    require(all(int(block['nx'+str(i+1)]) == n[i] for i in range(3)), 'Block input/header mismatch')
    require(np.all(ns%n == 0) and np.all(hi>lo), 'Invalid root tiling')
    require(np.array_equal(lo,h['region'][:3]) and np.array_equal(hi,h['region'][3:6]) and np.array_equal((hi-lo)/ns,h['region'][6:]), 'Domain input/header mismatch')
    require(h['root_level'] == math.ceil(math.log2(int(max(ns//n)))), 'Unexpected root logical level')
    shape = (nz+2*ng,ny+2*ng,nx+2*ng); cells = math.prod(shape) if hasattr(math,'prod') else int(np.prod(shape))
    faces = (shape[2]+1)*shape[1]*shape[0]+shape[2]*(shape[1]+1)*shape[0]+shape[2]*shape[1]*(shape[0]+1)
    offset = 8*(5*cells+faces)
    require(h['stride'] == offset+8*25*cells, 'Unsupported payload stride/precision')
    require(np.all(np.isfinite(h['costs'])) and np.all(h['costs'] == 1), 'Requires equal unit costs')
    return ng,n,ns,lo,hi,shape,offset,spin


def mesh_leaves(h,n,ns,lo,hi):
    leaves = []; locations = set()
    for gid,loc in enumerate(h['locations']):
        lx,ly,lz,level = map(int,loc); relative = level-h['root_level']
        require(0 <= relative <= 20 and tuple(loc) not in locations, 'Invalid/duplicate logical leaf')
        locations.add(tuple(loc)); logical = np.array([lx,ly,lz])
        require(np.all(logical >= 0) and np.all(logical < (ns//n)*2**relative), 'Leaf outside root domain')
        dx = (hi-lo)/ns/2**relative; lower = lo+logical*n*dx
        leaves.append(dict(gid=gid,relative_level=relative,logical_level=level,logical_location=logical.tolist(),lower=lower,dx=dx))
    volume = sum(float(np.prod(n*b['dx'])) for b in leaves)
    require(abs(volume-float(np.prod(hi-lo))) <= 1e-12*float(np.prod(hi-lo)), 'Leaf volume does not cover domain')
    for lx,ly,lz,level in locations:
        for ancestor in range(h['root_level'],level):
            shift = level-ancestor
            require((lx>>shift,ly>>shift,lz>>shift,ancestor) not in locations, 'Overlapping parent/child leaves')
    return leaves


def validate(run,ranks,cycle=None,exact_zero=False,outer_width=None):
    run = Path(run); candidates = list((run/'rst/rank_00000000').glob('*.rst'))
    require(candidates,'No rank0 checkpoint')
    headers = [read_header(p) for p in candidates]
    if cycle is None: cycle = max(h['cycle'] for h in headers)
    choices = [h for h in headers if h['cycle'] == cycle]
    require(len(choices) == 1,'Missing/ambiguous checkpoint cycle'); h = choices[0]
    ng,n,ns,lo,hi,shape,offset,spin = supported(h)
    blocks = mesh_leaves(h,n,ns,lo,hi); owners,counts = equal_partition(h['total'],ranks)
    name = Path(h['path']).name
    files = [run/'rst'/('rank_%08d'%rank)/name for rank in range(ranks)]
    require(set((run/'rst').glob('rank_*/'+name)) == set(files),'Missing/extra rank-file cohort')
    width = float(outer_width) if outer_width is not None else 4*float(np.max((hi-lo)/ns))
    require(math.isfinite(width) and width>0,'Invalid outer-face band width')
    horizon = math.sqrt(1-spin*spin); excise_chi = float(h['params']['z4c'].get('excise_chi',.0625))
    regions = {key:dict(proper_volume=0.,integrated_Theta_squared=0.,peak=None) for key in
               ('all','horizon_interior','horizon_exterior','r_gt_2M','outer_face_band','history_exterior')}
    minima = dict.fromkeys(('alpha','chi','gxx','second_minor','determinant'),float('inf'))
    invalid_count = nonzero = nonzero_active = 0; samples = []; records = []
    maxima = np.zeros(25); active_maxima = np.zeros(25); finite_all = True; gid0 = 0
    active_slice = (slice(ng,ng+int(n[2])),slice(ng,ng+int(n[1])),slice(ng,ng+int(n[0])))
    for rank,path in enumerate(files):
        before = path.stat()
        hr = read_header(path)
        require(hr['header'] == h['header'],'Rank%d header differs'%rank)
        require(path.stat().st_size == h['payload_start']+counts[rank]*h['stride'],'Rank%d truncated/extra payload or incorrect block ownership count'%rank)
        payload = np.memmap(path,dtype='<f8',mode='r',offset=h['payload_start'],shape=(counts[rank],h['stride']//8))
        finite = bool(np.isfinite(payload).all()); finite_all &= finite
        for local in range(counts[rank]):
            gid = gid0+local; b = blocks[gid]
            require(int(owners[gid]) == rank,'Contiguous gid ownership mismatch')
            u = payload[local,offset//8:].reshape((25,)+shape); ua = u[(slice(None),)+active_slice]
            nonzero += int(np.count_nonzero(u)); nonzero_active += int(np.count_nonzero(ua))
            maxima = np.maximum(maxima,np.max(abs(u),axis=(1,2,3)))
            active_maxima = np.maximum(active_maxima,np.max(abs(ua),axis=(1,2,3)))
            axes = [b['lower'][a]+(np.arange(int(n[a])+2*ng)-ng+.5)*b['dx'][a] for a in range(3)]
            z,y,x = np.meshgrid(axes[2],axes[1],axes[0],indexing='ij')
            r,alpha,chi,g = background(x,y,z,spin)
            alpha = alpha+u[18]; chi = chi+u[0]; g = [v+u[1+i] for i,v in enumerate(g)]
            xx,xy,xz,yy,yz,zz = g
            values = dict(alpha=alpha,chi=chi,gxx=xx,second_minor=xx*yy-xy*xy,determinant=determinant(g))
            invalid = np.zeros(shape,dtype=bool)
            for key,value in values.items():
                invalid |= ~np.isfinite(value)|(value<=0)
                minima[key] = min(minima[key],float(np.min(value)))
            invalid_count += int(np.count_nonzero(invalid))
            for kk,jj,ii in np.argwhere(invalid)[:max(0,16-len(samples))]:
                ijk = [int(ii),int(jj),int(kk)]
                samples.append(dict(rank=rank,gid=gid,relative_level=b['relative_level'],array_ijk=ijk,
                    xyz_M=[float(axes[a][ijk[a]]) for a in range(3)],
                    ghost_depth=[max(ng-ijk[a],ijk[a]-(ng+int(n[a])-1),0) for a in range(3)],
                    values={key:float(v[kk,jj,ii]) for key,v in values.items()}))
            if not finite or invalid.any(): continue
            rr = r[active_slice]; theta = u[17][active_slice]
            weight = np.sqrt(values['determinant'][active_slice])*chi[active_slice]**(-1.5)*float(np.prod(b['dx']))
            xa,ya,za = x[active_slice],y[active_slice],z[active_slice]
            face = np.minimum.reduce([xa-lo[0],hi[0]-xa,ya-lo[1],hi[1]-ya,za-lo[2],hi[2]-za])
            masks = dict(all=np.ones(theta.shape,dtype=bool),horizon_interior=rr<=horizon,horizon_exterior=rr>horizon,
                         r_gt_2M=rr>2,outer_face_band=(face<=width)&(rr>horizon),
                         history_exterior=(rr>horizon)&(chi[active_slice]>=excise_chi))
            for key,mask in masks.items():
                if not mask.any(): continue
                q = regions[key]; q['proper_volume'] += float(weight[mask].sum())
                q['integrated_Theta_squared'] += float((weight[mask]*theta[mask]**2).sum())
                magnitude = np.where(mask,abs(theta),-1); at = np.unravel_index(magnitude.argmax(),theta.shape)
                value = float(magnitude[at])
                if q['peak'] is None or value>q['peak']['absolute_Theta']:
                    k,j,i = map(int,at)
                    q['peak'] = dict(absolute_Theta=value,signed_Theta=float(theta[at]),rank=rank,gid=gid,
                         local_block=local,logical_level=b['logical_level'],relative_level=b['relative_level'],
                         xyz_M=[float(axes[0][i+ng]),float(axes[1][j+ng]),float(axes[2][k+ng])],radius_M=float(rr[at]))
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for data in iter(lambda:stream.read(8*1024*1024),b''): sha.update(data)
        after = path.stat()
        require((before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns), 'Checkpoint changed during audit: '+str(path))
        records.append(dict(rank=rank,path=str(path),bytes=after.st_size,sha256=sha.hexdigest(),
                            blocks=counts[rank],gid_first=gid0,gid_last=gid0+counts[rank]-1,all_payload_finite=finite))
        gid0 += counts[rank]; del payload
    for q in regions.values():
        q['proper_volume_Theta_RMS'] = math.sqrt(q['integrated_Theta_squared']/q['proper_volume']) if q['proper_volume'] else None
    return dict(passed=bool(finite_all and invalid_count==0 and (not exact_zero or nonzero==0)),
        scope=__doc__,time_M=h['time'],cycle=h['cycle'],checkpoint_dt=h['checkpoint_dt'],spin=spin,
        blocks=h['total'],ranks=ranks,blocks_per_rank=counts,relative_levels=sorted(set(b['relative_level'] for b in blocks)),
        common_header_sha256=h['header_sha256'],matching_headers=True,expected_payload_sizes=True,equal_unit_costs=True,
        ownership_scope='Implicit contiguous gid partition inferred from common mesh table, equal costs and exact rank-file lengths; format has no embedded per-block gid/rank signatures.',
        all_payload_finite=finite_all,invalid_metric_cells_including_ghosts=invalid_count,invalid_metric_samples=samples,
        raw_full_minima=minima,residual_exactly_zero=nonzero==0,exact_zero_required=exact_zero,
        residual_nonzero_count_including_ghosts=nonzero,residual_nonzero_count_active=nonzero_active,
        residual_field_max_including_ghosts=maxima.tolist(),residual_field_max_active=active_maxima.tolist(),
        background_reconstruction='Independent NumPy DBM stationary Kerr values plus conformal metric unit-determinant projection; last-bit equality to C++ is not claimed. No floors/repairs.',
        horizon_coordinate_radius_M=horizon,outer_face_band_width_M=width,history_excision_chi=excise_chi,
        theta_regions=regions,theta_regions_scope='Overlapping active-cell regions, physical proper-volume weights. History exterior also applies chi threshold. Invalid blocks/ranks excluded and force failure. Maxima do not identify first injection; no H/M constraints or stability claim.',files=records)


def safe_json(value):
    if isinstance(value,float) and not math.isfinite(value): return str(value)
    if isinstance(value,dict): return {k:safe_json(v) for k,v in value.items()}
    if isinstance(value,list): return [safe_json(v) for v in value]
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path); p.add_argument('--ranks',type=int,required=True)
    p.add_argument('--cycle',type=int); p.add_argument('--exact-zero',action='store_true')
    p.add_argument('--outer-width',type=float); p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    try: result = validate(a.run,a.ranks,a.cycle,a.exact_zero,a.outer_width)
    except Exception as exc: result = dict(passed=False,validation_error=str(exc),error_type=type(exc).__name__)
    result = safe_json(result); a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','residual_field_max_including_ghosts','residual_field_max_active')},indent=2,allow_nan=False))
    raise SystemExit(0 if result['passed'] else 1)
