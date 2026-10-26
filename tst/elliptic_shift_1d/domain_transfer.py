"""Explicit AF-to-finite-domain transfer; never a checkpoint-binding bypass.

Copies every retained stored field exactly in the original chart. Optional
positive target charts preserve physical curvature/Theta/Lambda bitwise and
record the invertible log conversion error. A new domain has its own source
binding and receipt. Scientific admission additionally requires the original
full AF prefix, convergence, cut-location and boundary controls.
"""
import hashlib,io,json
from pathlib import Path
import numpy as np
from gauge import lapse_coefficient

def load_af_transfer(args):
    path=Path(args.transfer_af_profile).resolve();manifest_path=path.parent/'manifest.json'
    manifest_bytes=manifest_path.read_bytes();source=json.loads(manifest_bytes);old=source['arguments']
    if source['status']!='finished' or not old.get('puncture_chart') or old.get('inner_radius',0.) or old.get('scalar'):
        raise ValueError('Transfer requires a completed full AF vacuum trajectory')
    if old.get('compensated_update',False):raise ValueError('A profile lacks the compensated integrator low part; this transfer is unsupported')
    equations=['backend','initial_data','mass','kappa1','kappa2','theta','damping_form','eta','slicing','slicing_family','slicing_scale',
      'lapse_reference','lapse_restoring','lapse_restoring_form','lapse_restoring_profile','lapse_background_source']
    for key in equations:
        if getattr(args,key)!=old[key]:raise ValueError('Domain transfer changes an equation/physical input: '+key)
    target_chart=args.puncture_chart or (args.metric_variable in ('log-trumpet','log-inner-trumpet') and args.lapse_variable=='log-reference')
    if not (args.puncture and target_chart and args.inner_radius>0 and not args.scalar and args.initial_data=='wormhole'):
        raise ValueError('Transfer requires the same vacuum wormhole in a supported positive chart on a physical cutoff')
    raw=path.read_bytes()
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:
        r=z['r'].copy();q=z['q'].copy();physical=z['q_physical'].copy();diagnostic=z['diagnostic'].copy();beta=z['beta'].copy()
        t=float(z['t']);step=int(z['step']);lattice=json.loads(str(z['time_lattice']))
    if r[0]!=0 or q.shape!=(len(r),8) or not np.all(np.isfinite(q)) or np.any(q[:,[1,7]]!=0):raise ValueError('Invalid full AF state')
    if t<=0 or t>=args.t or t>source['time']+1e-12:raise ValueError('Invalid domain transfer time/target')
    matches=np.flatnonzero(r==args.inner_radius)
    if len(matches)!=1 or matches[0]==0:raise ValueError('Cutoff must equal a saved physical grid node')
    cut=int(matches[0]);grid=r[cut:];state=q[cut:].copy()
    if len(grid)!=args.n or not np.array_equal(grid,np.linspace(args.inner_radius,1,args.n)):
        raise ValueError('Retained domain must be the exact saved uniform-grid subset')
    alpha=diagnostic[:,7];chi=diagnostic[:,8]
    if min(alpha[cut:])<=0 or min(chi[cut:])<=0 or not np.all(np.isfinite(diagnostic[cut:,:10])):raise ValueError('Invalid retained physical state')
    F=lapse_coefficient(grid[0],alpha[cut],args.slicing,args.slicing_family,args.slicing_scale)
    light=float(-beta[cut]+alpha[cut]*np.sqrt(chi[cut]));gauge=float(-beta[cut]+np.sqrt(F*chi[cut]))
    if max(light,gauge)>=0:raise ValueError('Cutoff has an incoming or tangent physical/gauge characteristic')
    if args.cfl*(grid[1]-grid[0])!=lattice['dt']:raise ValueError('Transfer must preserve the saved timestep lattice')
    ignored={'library','fast_library','output','resume','transfer_af_profile','report_dt','t'}
    changes={key:dict(old=old.get(key),new=value) for key,value in vars(args).items() if key not in ignored and old.get(key)!=value}
    receipt=dict(profile=str(path),profile_sha256=hashlib.sha256(raw).hexdigest(),source_manifest=str(manifest_path),
        source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),source_binding=source['binding'],time=t,step=step,
        original_grid_points=len(r),dropped_grid_points=cut,retained_grid_points=len(grid),inner_radius=float(grid[0]),
        stored_fields_copied_bitwise=bool(args.puncture_chart),target_metric_variable=args.metric_variable,target_lapse_variable=args.lapse_variable,target_AF_chart=bool(args.puncture_chart),equation_inputs_verified=equations,numerical_input_changes=changes,
        source_physical_characteristic=light,source_gauge_characteristic=gauge,
        inner_elliptic_condition='Inherited Q=0 metric-freezing branch; outer null shift fixes phase. No differential Einstein constraint enters.',
        scope='Domain transition only; original full-prefix convergence and later stage characteristic/boundary/physical checks remain mandatory. No wormhole-to-trumpet admission.')
    return dict(q=state,r=grid,t=t,step=step,lattice=lattice,receipt=receipt,physical=physical[cut:],chi=chi[cut:],alpha=alpha[cut:],beta=beta[cut:])
