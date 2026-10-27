#!/usr/bin/env python3
"""Read-only completed3D/RK4 family comparison; differing meshes imply no order fit."""
import argparse
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np

PROFILE_GRID_NOTE='different meshes'
FIELDS=('alpha','chi','g_rr','A_rr','Khat','Theta','Lambda_r','beta_r')
CASES=(
 ('three_shell_n8','N8:3 shells, enriched','enriched-buffered-pilot-20261027','three-shell.csv','three-shell.stdout','commands.json',[.62,.70,1]),
 ('seven_shell_n8','N8:7 shells, radial lines','enriched-line-pilot-20261027','seven-shell.csv','pilot.stdout','command.json',[.45,.55,.60,.65,.70,.75,1]),
 ('amr_core_n6','N6:7 shells, core AMR3','amr-core-pilot-20261027','coupled.csv','coupled.stdout','coupled-command.json',[.45,.55,.60,.65,.70,.75,1]))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def interpolate(r,values,x,width=7):
    x=np.asarray(x);first=np.clip(np.searchsorted(r,x)-width//2,0,len(r)-width);result=np.zeros_like(x)
    for i in range(width):
        weight=np.ones_like(x)
        for j in range(width):
            if i!=j:weight*=(x-r[first+j])/(r[first+i]-r[first+j])
        result+=weight*values[first+i]
    return result


def norms(error,weights):
    error=np.asarray(error);scale=float(np.max(abs(error)));volume=float(np.sum(weights))
    square=float(np.sum(weights*(error/scale)**2)) if scale else 0.
    return dict(penrose_volume_rms=scale*float(np.sqrt(square/volume)),point_max=scale,
                weighted_l2=scale*float(np.sqrt(square)),signed_bias=float(np.sum(weights*error)/volume))


def topology(data,n,radii):
    count=len(data);size=n**3
    if count%size:raise ValueError('active-cell block shape mismatch')
    xyz=np.column_stack([data[x] for x in ('x','y','z')]);block=np.arange(count)//size
    i=np.arange(count)%n;j=np.arange(count)//n%n;k=np.arange(count)//(n*n)%n
    sides=((i==0)|(i==n-1)).astype(int)+((j==0)|(j==n-1)).astype(int)+((k==0)|(k==n-1)).astype(int)
    cart=np.max(abs(xyz),axis=1)<=.25*(1+2e-14);blocks=[];spacing=np.full(count,np.nan);levels=np.full(count,-1,dtype=int)
    for b in range(count//size):
        select=slice(b*size,(b+1)*size);q=xyz[select]
        if np.all(cart[select]):
            h=np.array([q[1,0]-q[0,0],q[n,1]-q[0,1],q[n*n,2]-q[0,2]])
            if not np.all(h>0) or not np.allclose(h,h[0],rtol=1e-12,atol=1e-14):raise ValueError('Cartesian cell spacing/axis mismatch')
            lo=q.min(axis=0)-h/2;hi=q.max(axis=0)+h/2;level=int(round(np.log2((.25/n)/h[0])));spacing[select]=h[0];levels[select]=level
            blocks.append(dict(id=b,cartesian=True,lo=lo,hi=hi,h=h[0],level=level))
        elif np.any(cart[select]):raise ValueError('mixed Cartesian/shell block')
        else:
            layer=int(np.searchsorted(radii,float(data['r'][select].max())));blocks.append(dict(id=b,cartesian=False,layer=layer))
    cf=np.zeros(count,dtype=bool);cart_shell=np.zeros(count,dtype=bool);shell_radial=np.zeros(count,dtype=bool);pairs=[]
    indices=np.column_stack([i,j,k]);cartblocks=[b for b in blocks if b['cartesian']]
    for b in cartblocks:
        s=slice(b['id']*size,(b['id']+1)*size)
        for a in range(3):
            if abs(b['lo'][a]+.25)<1e-13:cart_shell[s]|=indices[s,a]<2
            if abs(b['hi'][a]-.25)<1e-13:cart_shell[s]|=indices[s,a]>=n-2
    for index,b in enumerate(cartblocks):
        for c in cartblocks[index+1:]:
            if abs(b['h']-c['h'])<1e-13:continue
            for a in range(3):
                if abs(b['hi'][a]-c['lo'][a])<1e-13:side_b,side_c=1,0
                elif abs(c['hi'][a]-b['lo'][a])<1e-13:side_b,side_c=0,1
                else:continue
                tang=[d for d in range(3) if d!=a];low=np.maximum(b['lo'][tang],c['lo'][tang]);high=np.minimum(b['hi'][tang],c['hi'][tang])
                if np.any(high-low<1e-13):continue
                pairs.append(dict(blocks=[b['id'],c['id']],axis=a,coordinate=float(b['hi'][a] if side_b else b['lo'][a]),levels=[b['level'],c['level']]))
                for q,side in [(b,side_b),(c,side_c)]:
                    s=slice(q['id']*size,(q['id']+1)*size);m=(indices[s,a]>=n-2) if side else (indices[s,a]<2)
                    for d,lo,hi in zip(tang,low,high):m&=(xyz[s,d]>lo-1e-13)&(xyz[s,d]<hi+1e-13)
                    cf[s]|=m
    for b in blocks:
        if b['cartesian']:continue
        s=slice(b['id']*size,(b['id']+1)*size)
        if b['layer']==0:cart_shell[s]|=i[s]<2
        if b['layer']>0:shell_radial[s]|=i[s]<2
        if b['layer']<len(radii)-1:shell_radial[s]|=i[s]>=n-2
    masks=dict(all=np.ones(count,dtype=bool),puncture_r_lt_005=data['r']<.05,core_r_lt_01=data['r']<.1,
        bulk_r_01_03=(data['r']>=.1)&(data['r']<.3),transition_r_03_08=(data['r']>=.3)&(data['r']<.8),focused_r_062_070=(data['r']>=.62)&(data['r']<.70),outer_r_ge_08=data['r']>=.8,scri_r_ge_09=data['r']>=.9,
        cartesian=cart,shell=~cart,block_interface_neighbors=sides>0,block_edge_corner_neighbors=sides>1,cartesian_shell_stencil_neighbors=cart_shell,radial_shell_stencil_neighbors=shell_radial,cartesian_coarse_fine_stencil_neighbors=cf,
        enrichment_geometric_support=cart&(data['r']<.1)&(data['r']/spacing<=8))
    for level in sorted(set(levels[cart])):masks[f'cartesian_level_{level}']=levels==level
    for layer in range(len(radii)):masks[f'shell_layer_{layer}']=np.array([not b['cartesian'] and b['layer']==layer for b in blocks])[block]
    return masks,dict(block=block,i=i,j=j,k=k,spacing=spacing,levels=levels,blocks=blocks,coarse_fine_pairs=pairs)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,default=Path('docs/elliptic-shift-3d/evidence'));parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();inputs={}
    def track(path):inputs[str(path)]=digest(path);return path
    def csv(path):
        path=path if path.exists() else path.with_name(path.name+'.gz');track(path);a=np.genfromtxt(path,delimiter=',',names=True)
        if not len(a) or not all(np.all(np.isfinite(a[f])) for f in a.dtype.names):raise ValueError(f'Empty/nonfinite input; no filtering: {path}')
        return a
    refroot=args.evidence/'selected-taper-pilot-20261027';ref=np.load(track(refroot/'radial-n2049.npz'),allow_pickle=False);coarse=np.load(track(refroot/'radial-n1025.npz'),allow_pickle=False);parameters=json.loads(track(refroot/'reference-parameters.json').read_text())
    if parameters['telegraph_width']!=.24 or parameters['time']!=.0002 or parameters['reference']!='Minkowski':raise ValueError('reference is not selected .24 Minkowski t=.0002')
    summary=dict(scope='Three genuine3D/RK4 evolutions at t=.0002, diagnostic angular averages only; all scientific FAIL; differing meshes/solvers/enrichment imply no uniform-resolution convergence claim',reference_parameters=parameters,
        weight_definition='w_p=map_cell_volume*sqrt(det(g_tilde))/chi^(3/2), Penrose spatial metric gamma_bar=g_tilde/chi. No additional Omega^(-3), floor, clipping, radial exclusion or discarded active cell. RMS=sqrt(sum(w e²)/sum(w)); unnormalized L2 also retained.',
        vector_norm_definition='For beta vector errors and tangential beta, Euclidean norm of common Cartesian components, then Penrose-volume weighting; not metric-length weighting.',
        region_note='all includes every active cell. Other regions overlap and are diagnostics only. Two active-cell layers mark fourth-order stencil neighbors of inferred Cartesian coarse-fine or patch interfaces.',
        unavailable='CSV lacks metric derivatives and local original elliptic residuals. Regional Z and regional original shift residuals cannot be reconstructed faithfully. Retain stdout global Z and original-equation stage maxima; do not substitute independent Lambda for Z.',cases={})
    arrays={}
    for key,label,folder,filename,stdout_name,command_name,radii in CASES:
        base=args.evidence/folder;result=json.loads(track(base/stdout_name).read_text());commands=json.loads(track(base/command_name).read_text());data=csv(base/filename);spheres=csv(base/(filename+'.spheres.csv'))
        if result['time']!=.0002 or result['telegraph_width']!=.24 or result['positive_storage']!=2 or result['cells']!=len(data):raise ValueError('case/reference/physical chart mismatch')
        if not np.all(data['weight']>0) or not np.all(data['chi']>0) or not np.all(data['alpha']>0):raise ValueError('nonpositive physical values; no floor or filter')
        if not np.all((data['r']>0)&(data['r']<=1)):raise ValueError('unsupported endpoint/radius; no exclusion')
        w=data['weight'];masks,mesh=topology(data,result['n'],radii);truth={f:interpolate(ref['r'],ref[f],data['r']) for f in FIELDS};errors={f:data[f]-truth[f] for f in FIELDS};ref_error={f:interpolate(coarse['r'],coarse[f],data['r'])-truth[f] for f in FIELDS}
        coords=np.column_stack([data[f] for f in ['x','y','z']]);normal=coords/data['r'][:,None];beta=np.column_stack([data['delta_beta_'+f]-data[f] for f in ['x','y','z']]);beta_error=beta-truth['beta_r'][:,None]*normal;beta_vector_error=np.linalg.norm(beta_error,axis=1);beta_tangent=np.linalg.norm(beta-np.sum(beta*normal,axis=1)[:,None]*normal,axis=1)
        source_exit=1
        if isinstance(commands,list):matching=[c for c in commands if any(filename==Path(str(a)).name for a in c.get('command',[]))];source_exit=matching[-1].get('exit',matching[-1].get('exit_code')) if matching else None
        else:source_exit=commands.get('exit',commands.get('exit_code'))
        row=dict(label=label,recorded_scientific_status='FAIL',recorded_exit=source_exit,reported=result,commands=commands,all_cells=len(data),minimum_radius=float(data['r'].min()),penrose_volume_quadrature=float(w.sum()),
            field_errors={f:norms(errors[f],w) for f in FIELDS},reference_n1025_minus_n2049={f:norms(ref_error[f],w) for f in FIELDS},regions={},angular={},global_Z=dict(reported_penrose_covector_norm_rms=result['Z_rms'],regional_available=False,reason='Requires metric-connection derivatives absent from CSV'),
            original_shift=dict(source='Reported original-equation stage maxima; not recomputed from derivative-incomplete CSV',physical_rms_max=result['shift_residual_max'],point_max=result['shift_residual_point_max']),maxima={},
            enriched_support_count=int(masks['enrichment_geometric_support'].sum()),coarse_fine_face_pairs=mesh['coarse_fine_pairs'])
        if row['enriched_support_count']!=result['enriched_cells']:raise ValueError('inferred enrichment support does not match recorded case')
        for name,mask in masks.items():
            if not mask.any():row['regions'][name]=dict(cells=0,unavailable=True);continue
            ww=w[mask];row['regions'][name]=dict(cells=int(mask.sum()),penrose_volume=float(ww.sum()),volume_fraction=float(ww.sum()/w.sum()),H=norms(data['H'][mask],ww),M=norms(data['M_norm'][mask],ww),Theta=norms(data['Theta'][mask],ww),
                field_errors={f:norms(errors[f][mask],ww) for f in FIELDS},beta_vector_error=norms(beta_vector_error[mask],ww),beta_tangent=norms(beta_tangent[mask],ww),Z=dict(available=False),original_shift_residual=dict(available=False))
        for f,column in [('H','H'),('M','M_norm'),('Theta','Theta')]:
            recomputed=row['regions']['all'][f]['penrose_volume_rms'];reported=result[f+'_rms']
            if abs(recomputed-reported)>5e-12*max(1,abs(reported)):raise ValueError('CSV fails to reproduce reported all-domain '+f)
        for name,values in [('H',data['H']),('M',data['M_norm']),('Theta',data['Theta']),('beta_radial_error',errors['beta_r']),('beta_vector_error',beta_vector_error)]:
            p=int(abs(values).argmax());b=int(mesh['block'][p]);h=mesh['spacing'][p];row['maxima'][name]=dict(point=p,xyz=coords[p].tolist(),radius=float(data['r'][p]),signed_value=float(values[p]),absolute_value=float(abs(values[p])),block=b,logical_ijk=[int(mesh[a][p]) for a in ['i','j','k']],cartesian_level=int(mesh['levels'][p]),cartesian_h=float(h) if np.isfinite(h) else None,
                rho_over_h=float(data['r'][p]/h) if np.isfinite(h) else None,in_enrichment_geometric_support=bool(masks['enrichment_geometric_support'][p]),in_coarse_fine_stencil_neighbor=bool(masks['cartesian_coarse_fine_stencil_neighbors'][p]))
        for f in FIELDS:
            rs=np.unique(spheres['r']);means=[];sd=[];small=[];large=[];rms=[];raw=[]
            exact=interpolate(ref['r'],ref[f],rs);reference_control=interpolate(coarse['r'],coarse[f],rs)-exact
            for radius,value in zip(rs,exact):
                select=spheres['r']==radius;directions=spheres['direction'][select];v=spheres[f][select]
                if len(v)!=64 or len(np.unique(directions))!=64:raise ValueError('angular64 raw sampling mismatch')
                means.append(float(v.mean()));sd.append(float(v.std()));small.append(float(v.min()));large.append(float(v.max()));rms.append(float(np.sqrt(np.mean((v-value)**2))));raw.append(v)
            row['angular'][f]=dict(radii=rs.tolist(),directions_per_radius=64,reference=exact.tolist(),mean=means,raw_scatter=sd,minimum=small,maximum=large,mean_error=(np.array(means)-exact).tolist(),angular_rms_error=rms,reference_n1025_minus_n2049=reference_control.tolist(),maximum_scatter=max(sd),variance_identity_max_error=float(np.max(abs(np.array(rms)**2-(np.array(means)-exact)**2-np.array(sd)**2))))
        row['gates']={name:dict(value=result[name],unchanged_target=tol,passes=result[name]<=tol) for name,tol in [('scri_tensor_pole',1e-7),('scri_ghost_clock',1e-13),('scri_ghost_null',1e-12),('scri_ghost_tangent',1e-12)]}
        np.savez_compressed(out/(key+'-all-cell-errors.npz'),points=coords,radius=data['r'],weights=w,fields=np.array(FIELDS),physical_fields=np.column_stack([data[f] for f in FIELDS]),reference=np.column_stack([truth[f] for f in FIELDS]),errors=np.column_stack([errors[f] for f in FIELDS]),reference_n1025_minus_n2049=np.column_stack([ref_error[f] for f in FIELDS]),H=data['H'],M=data['M_norm'],Theta=data['Theta'],beta=beta,beta_vector_error=beta_error,beta_tangent=beta_tangent,cartesian_levels=mesh['levels'],cartesian_spacing=mesh['spacing'])
        np.savez_compressed(out/(key+'-raw-angular.npz'),radius=spheres['r'],direction=spheres['direction'],fields=np.array(FIELDS),values=np.column_stack([spheres[f] for f in FIELDS]))
        summary['cases'][key]=row;arrays[key]=(data,mesh)
    # Preserve existing localization metadata as provenance and explicitly compare
    # against the actual t=.0002 CSV rather than treating a nearby t0 value as final.
    location=args.evidence/'amr-core-pilot-20261027/maximum-H-localization.json'
    if location.exists():summary['prior_AMR_localization_record']=json.loads(track(location).read_text())
    control=args.evidence/'enriched-line-pilot-20261027/independent-controls.json'
    if control.exists():summary['reported_independent_controls']=dict(record=json.loads(track(control).read_text()),units='Maximum stored-field/shift differences and H differences; not substituted for physical-field RMS controls')
    summary['input_hashes']=inputs
    with tempfile.TemporaryDirectory(prefix='plot-cache-',dir=out) as cache:
        os.environ['MPLCONFIGDIR']=cache;os.environ['XDG_CACHE_HOME']=cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors=['tab:blue','tab:orange','tab:green']
        fig,axes=plt.subplots(4,2,figsize=(11,12),layout='constrained')
        for ax,f in zip(axes.flat,FIELDS):
            ax.plot(ref['r'],ref[f],color='k',lw=1.1,label='Matched selected .24 radial N2049')
            for (key,row),color in zip(summary['cases'].items(),colors):
                a=row['angular'][f];r=np.array(a['radii']);mean=np.array(a['mean']);scatter=np.array(a['raw_scatter'])
                ax.plot(r,mean,'.-',ms=3,color=color,label=row['label']);ax.fill_between(r,mean-scatter,mean+scatter,color=color,alpha=.13)
            ax.set(xlabel='compactified radius',ylabel=f);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=6.5);fig.suptitle(f'Coupled 3D/RK4 at t=0.0002: all scientific FAIL\nAngular means +/- raw scatter; {PROFILE_GRID_NOTE}, no convergence-order claim')
        fig.savefig(out/'physical-profiles.png',dpi=155);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');x=np.arange(len(FIELDS))
        for (key,row),color in zip(summary['cases'].items(),colors):
            axes[0,0].semilogy(x,[row['field_errors'][f]['penrose_volume_rms'] for f in FIELDS],'.-',color=color,label=row['label'])
            axes[0,0].semilogy(x,[row['reference_n1025_minus_n2049'][f]['penrose_volume_rms'] for f in FIELDS],':',color=color)
            axes[0,1].semilogy(x,[row['field_errors'][f]['point_max'] for f in FIELDS],'.-',color=color)
            axes[1,0].semilogy(x,[row['angular'][f]['maximum_scatter'] for f in FIELDS],'.-',color=color)
            axes[1,1].semilogy(x,[max(abs(np.array(row['angular'][f]['mean_error']))) for f in FIELDS],'.-',color=color)
        for ax in axes.flat:ax.set_xticks(x,FIELDS,rotation=35,ha='right');ax.grid(alpha=.2)
        axes[0,0].set_title('All-cell Penrose-volume RMS; dotted: radial N1025-N2049');axes[0,0].legend(fontsize=7)
        axes[0,1].set_title('All-cell maximum physical-field error');axes[1,0].set_title('Maximum raw angular scatter over sampled radii');axes[1,1].set_title('Maximum angular-mean error over sampled radii')
        fig.suptitle('Physical error budget against matched .24 radial reference\nNo RMS exclusions, no uniform-resolution convergence inference')
        fig.savefig(out/'physical-error-budget.png',dpi=155);plt.close(fig)
        names=['all','puncture_r_lt_005','core_r_lt_01','bulk_r_01_03','transition_r_03_08','focused_r_062_070','outer_r_ge_08','scri_r_ge_09','cartesian_coarse_fine_stencil_neighbors','cartesian_shell_stencil_neighbors','radial_shell_stencil_neighbors']
        labels=['all','r<.05','r<.1','.1-.3','.3-.8','.62-.70','r>=.8','r>=.9','coarse-fine','cube-shell','shell radial']
        fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
        for ax,channel in zip(axes.flat[:3],['H','M','Theta']):
            for (key,row),color in zip(summary['cases'].items(),colors):
                values=[row['regions'][name].get(channel,{}).get('penrose_volume_rms',np.nan) for name in names];ax.semilogy(np.arange(len(names)),values,'.-',color=color,label=row['label'])
            ax.set_xticks(np.arange(len(names)),labels,rotation=45,ha='right');ax.set_title(channel+' RMS: full domain and overlapping regions');ax.grid(alpha=.2)
        ax=axes.flat[3]
        for (key,row),color in zip(summary['cases'].items(),colors):ax.semilogy([0,1,2],[row['reported']['Z_rms'],row['original_shift']['physical_rms_max'],row['reported']['scri_tensor_pole']],'.-',color=color,label=row['label'])
        ax.set_xticks([0,1,2],['global Z RMS','shift original RMS max','scri tensor pole'],rotation=20,ha='right');ax.set_title('Reported global diagnostics; local Z/shift residual unavailable');ax.grid(alpha=.2);axes[0,0].legend(fontsize=7)
        fig.suptitle('Constraint and gauge diagnostics: all scientific FAIL\nAll-cell norms reproduced from CSV; global Z/shift retained from stdout')
        fig.savefig(out/'constraint-regions.png',dpi=155);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for (key,row),color in zip(summary['cases'].items(),colors):
            data,mesh=arrays[key];axes[0].scatter(data['r'],abs(data['H']),s=2,alpha=.17,color=color,rasterized=True,label=row['label']);q=row['maxima']['H'];axes[0].plot(q['radius'],q['absolute_value'],'x',ms=8,color=color)
        axes[0].set_xscale('log');axes[0].set_yscale('symlog',linthresh=1e-8);axes[0].set_ylim(bottom=0);axes[0].set(xlabel='compactified radius (every active cell)',ylabel='|H|');axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
        # A wrapper may compare other families. Select the last case's maximum
        # CARTESIAN H for this diagnostic slice; global maxima/norms above
        # always retain every cell, including shell maxima.
        slice_key=next(reversed(arrays));data,mesh=arrays[slice_key]
        cartesian=np.flatnonzero(mesh['levels']>=0)
        point=int(cartesian[np.argmax(abs(data['H'][cartesian]))])
        q=dict(xyz=[float(data[a][point]) for a in ['x','y','z']],
               cartesian_h=float(mesh['spacing'][point]),point=point)
        z=q['xyz'][2];select=(abs(data['z']-z)<q['cartesian_h']*.51)&(np.max(np.column_stack([abs(data['x']),abs(data['y'])]),axis=1)<.13)
        plot=axes[1].scatter(data['x'][select],data['y'][select],c=abs(data['H'][select]),s=13,cmap='magma',rasterized=True)
        for b in mesh['blocks']:
            if b['cartesian'] and b['lo'][2]<=z<=b['hi'][2] and np.max(abs(b['lo'][:2]))<.14:
                axes[1].add_patch(plt.Rectangle(b['lo'][:2],b['hi'][0]-b['lo'][0],b['hi'][1]-b['lo'][1],fill=False,color='cyan',lw=.5,alpha=.6))
        axes[1].plot(q['xyz'][0],q['xyz'][1],'x',color='lime',ms=9);axes[1].set(xlim=(-.13,.13),ylim=(-.13,.13),xlabel='x',ylabel='y',title=f'Cartesian diagnostic slice z={z:.6g}; core max H node{q["point"]}')
        fig.colorbar(plot,ax=axes[1],label='|H|');fig.suptitle('Localization only: slice selection does not exclude any cell from norms')
        fig.savefig(out/'hamiltonian-localization.png',dpi=155);plt.close(fig)
        summary['software']=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__)
    summary['provenance']=dict(command=sys.argv,analysis_script_sha256=digest(Path(__file__)),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-start,inputs_unchanged=all(pathlib_exists_and_hash(Path(p),v) for p,v in inputs.items()))
    (out/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:dict(H_rms=row['reported']['H_rms'],M_rms=row['reported']['M_rms'],physical_volume=row['penrose_volume_quadrature'],alpha_error=row['field_errors']['alpha']['penrose_volume_rms'],beta_error=row['field_errors']['beta_r']['penrose_volume_rms'],H_maximum=row['maxima']['H']) for key,row in summary['cases'].items()}))


def pathlib_exists_and_hash(path,value):return path.exists() and digest(path)==value
if __name__=='__main__':main()
