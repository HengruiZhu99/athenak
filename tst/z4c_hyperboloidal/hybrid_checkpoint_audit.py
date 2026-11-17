#!/usr/bin/env python3
"""Reduce fixed native checkpoint slots; cross-check native history-band Theta.

All outputs are analysis artifacts in the requested new directory. Never evolves
or rewrites a checkpoint. Coordinate volume and the unchanged physical partition
are used; inactive Cartesian slots have no weight.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import numpy as np
from restart_roundtrip import parse, payload

EDGES = np.array([.18,.20,.25,.32,.40,.55,.80,.94,1.])


def sample_weights(nodes, target):
    """Six nominal nodes, with a one-sided stencil near a patch edge."""
    center = int(np.searchsorted(nodes, target))
    first = max(0, min(len(nodes)-6, center-3))
    index = np.arange(first, first+6)
    x = nodes[index]
    weights = np.ones(6)
    for i in range(6):
        for j in range(6):
            if i != j:
                weights[i] *= (target-x[j])/(x[i]-x[j])
    return index, weights

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def check_overlap_observer(history_dir, checkpoint, time, snapshot_dir=None):
    overlap_file=history_dir/'checkpoint.overlap.hst'
    overlap=np.atleast_2d(np.loadtxt(overlap_file))[-1]
    assert len(overlap)==10 and np.isfinite(overlap).all()
    assert abs(overlap[0]-time)<1e-12
    assert np.all(overlap[2:8]>=0) and overlap[8]==0 and overlap[9]>0
    observation={'file':str(overlap_file),'sha256':sha(overlap_file),
        'native_row':overlap.tolist(),
        'note':'Common retained-overlap observer with kappa=0; zero energy contribution is diagnostic-only.',
        'runtime_history_comparison':None}
    # Compare the independent native observer to the production observable
    # at this exact checkpoint time when that mode already writes one.
    for source in checkpoint.parent.parent.glob('*.overlap.hst'):
        source_bytes=source.read_bytes()
        values=np.atleast_2d(np.loadtxt(io.BytesIO(source_bytes)))
        matched=values[np.abs(values[:,0]-time)<1e-12]
        if len(matched):
            np.testing.assert_allclose(overlap[[2,3,4,5,6,7,9]],matched[-1,[2,3,4,5,6,7,9]],
                                       rtol=3e-12,atol=1e-28)
            observation['runtime_history_comparison']={
                'file':str(source),'sha256':hashlib.sha256(source_bytes).hexdigest(),
                'matched_native_row':matched[-1].tolist(),'rtol':3e-12,'passed':True}
            if snapshot_dir:
                snapshot=snapshot_dir/(source.name+'.runtime-snapshot')
                snapshot.write_bytes(source_bytes)
                observation['runtime_history_comparison']['snapshot_file']=str(snapshot)
    return observation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--history-dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    for source in [Path(__file__),Path(__file__).with_name('restart_roundtrip.py')]:
        (a.output/source.name).write_bytes(source.read_bytes())
    info=parse(a.checkpoint);identity=info['identity'].decode()
    atlas=dict(x.split('=',1) for x in info['atlas'].decode().split(';')[1:])
    assert 'radial_blocks' not in atlas, 'This legacy quadrature requires one block per radial region; subdivided shells need native mapped volumes'
    n=int(re.search(r';n=(\d+),',identity)[1]);o=int(re.search(r';overlap=(\d+)',identity)[1])
    hw=float.fromhex(atlas['half_width']);inner=float.fromhex(atlas['inner_radius'])
    radii=[inner]+[float.fromhex(x) for x in atlas['radii'].split(',') if x]
    core=[int(x) for x in atlas['core'].split(',') if x]
    assert core==[3,3,3] and atlas['angular']=='1,1,' and atlas['equiangular']=='1' and atlas['outer_endpoint']=='1'
    cutoff=float.fromhex(re.search(r';p.activity.cartesian_cutoff=([^;]+)',identity)[1]) if 'activity-policy=' in identity else np.inf
    with a.checkpoint.open('rb') as f:
        f.seek(info['start']+4);rootlevel=struct.unpack('=i',f.read(4))[0]
    locs=np.frombuffer(info['topology'],dtype='=i4').reshape(-1,5)
    data=payload(info);bulk=(n+2*o)**3
    stats=np.zeros((8,2));counts=np.zeros(3);all_counts=np.zeros(3);floor_radius=0.;shells=np.zeros((6,len(radii)-1,n,n,n))
    layer_stats=np.zeros((len(radii)-1,5));active_count=0
    angular=np.tan(np.pi/4*(-1+2*(np.arange(n)+.5)/n))
    corner=np.zeros((n,n),bool)
    # Fixed angular region, independent of resolution; not a fixed cell count.
    vv,uu=np.meshgrid(-1+2*(np.arange(n)+.5)/n,-1+2*(np.arange(n)+.5)/n,indexing='ij')
    corner=(abs(vv)>=.75)&(abs(uu)>=.75)
    for gid,loc in enumerate(locs):
        if not np.isfinite(data[gid]).all():raise ValueError('nonfinite full checkpoint block '+str(gid))
        all_counts+=data[gid,:,33:36].sum(axis=0)
        i,j,k,level,chart=map(int,loc);factor=2**(level-rootlevel)
        u=np.array(data[gid,:bulk].reshape(n+2*o,n+2*o,n+2*o,39)[o:o+n,o:o+n,o:o+n,:])
        if not np.isfinite(u).all():raise ValueError('nonfinite checkpoint block '+str(gid))
        if chart==0:
            dx=2*hw/(3*factor*n)
            coords=[-hw+(idx*n+np.arange(n)+.5)*dx for idx in (i,j,k)]
            z,y,x=np.meshgrid(coords[2],coords[1],coords[0],indexing='ij')
            r=np.sqrt(x*x+y*y+z*z);weight=np.full(r.shape,dx**3);active=r<cutoff
        else:
            assert factor==1 and j==0 and k==0, 'shell AMR needs a separate geometric quadrature'
            v,w,r=np.meshgrid(angular,angular,radii[i]+(np.arange(n)+(0 if i==len(radii)-2 else .5))*(radii[i+1]-radii[i])/n,indexing='ij')
            weight=r*r*(radii[i+1]-radii[i])/n*(np.pi/(2*n))**2*(1+w*w)*(1+v*v)/(1+w*w+v*v)**1.5
            if i==len(radii)-2:weight[:,:,0]*=.5
            active=r>=inner
        blend=np.clip((r-inner)/(min(hw,cutoff)-inner),0,1)
        blend=blend**3*(10+blend*(-15+6*blend))
        weight*=1-blend if chart==0 else blend
        weight*=active;theta=u[:,:,:,14]  # retain native units until plotting
        for b in range(8):
            mask=(r>=EDGES[b]-1e-12)&(r<EDGES[b+1]-1e-12)&(weight>0)
            stats[b,0]+=weight[mask].sum();stats[b,1]+=(weight[mask]*theta[mask]**2).sum()
        counts+=u[:,:,:,33:36][active].sum(axis=0);active_count+=int(active.sum())
        fl=(u[:,:,:,33]>0)&active
        if fl.any():floor_radius=max(floor_radius,float(r[fl].max()))
        if chart:
            shells[chart-1,i]=theta
            e=weight*theta**2
            layer_stats[i]+=[weight.sum(),e.sum(),weight[corner,:].sum(),e[corner,:].sum(),0]
            layer_stats[i,4]=max(layer_stats[i,4],float(abs(theta).max()))
    samples=[]
    normalized_angle=-1+2*(np.arange(n)+.5)/n
    for radius in [.34,.36,.38]:
        layer=int(np.searchsorted(radii,radius,side='right')-1)
        rr=radii[layer]+(np.arange(n)+(0 if layer==len(radii)-2 else .5))*(radii[layer+1]-radii[layer])/n
        ir,wr=sample_weights(rr,radius)
        for angle in [-.85,.85]:
            ia,wa=sample_weights(normalized_angle,angle)
            for face in range(6):
                values=shells[face,layer][np.ix_(ia,ia,ir)]
                samples.append({'face':face,'radius':radius,'u':angle,'v':angle,
                                'native_Theta':float(np.einsum('i,j,k,ijk',wa,wa,wr,values))})
    result={'checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':sha(a.checkpoint),
        'script_sha256':sha(Path(__file__)),'restart_parser_sha256':sha(Path(__file__).with_name('restart_roundtrip.py')),'identity':identity,'atlas':info['atlas'].decode(),
        'time':info['time'],'time_M0':3*info['time'],'cycle':info['cycle'],'n':n,
        'fixed_band_edges':EDGES.tolist(),'band_boundary_tolerance':1e-12,'fixed_band_columns':['volume','native_Theta_squared_integral'],
        'fixed_band_sums':stats.tolist(),'shell_radii':radii,'shell_columns':['volume','Theta2_integral','corner_volume','corner_Theta2_integral','max_abs_Theta'],
        'shell_sums':layer_stats.tolist(),'corner_definition':'both absolute normalized angular coordinates >= 0.75',
        'nominal_active_points':active_count,'nominal_cumulative_chi_counts':counts.tolist(),
        'chi_count_columns':['floor','nonpositive','nonfinite'],
        'all_evolved_and_scri_cumulative_chi_counts':all_counts.tolist(),
        'common_coordinate_samples':samples,'sample_interpolation':'six-point tensor Lagrange on nominal shell nodes; identical physical r,u,v across resolutions; diagnostic only',
        'max_radius_any_historical_floor':floor_radius,'native_history_check':None}
    if a.history_dir:
        checks=[]
        for b in range(8):
            f=a.history_dir/('checkpoint-band-'+str(b)+'.user.hst')
            row=np.atleast_2d(np.loadtxt(f))[-1]
            assert abs(row[0]-info['time'])<1e-12
            np.testing.assert_allclose(row[[2,5]],stats[b],rtol=3e-12,atol=1e-28)
            checks.append({'file':str(f),'sha256':sha(f),'native_raw_integrals':row[2:].tolist()})
        result['native_history_check']=checks
        result['native_checkpoint_overlap']=check_overlap_observer(a.history_dir,a.checkpoint,info['time'],a.output)
    np.savez_compressed(str(a.output/'shell-theta.npz'),theta=shells)
    (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS '+('checkpoint Theta/volume against native history' if a.history_dir else 'finite checkpoint reduction (no native-history comparison requested)'),info['cycle'],flush=True)

if __name__=='__main__':main()
