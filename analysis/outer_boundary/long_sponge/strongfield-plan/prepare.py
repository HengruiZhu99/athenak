"""Prepare static centered-trumpet inputs. This script never launches AthenaK."""
from pathlib import Path
import hashlib,json,re
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'fast-weakfield/input.athinput'
s=BASE.read_text()
def change(text,section,key,value):
 pat=r'(?ms)(<'+re.escape(section)+r'>\n)(.*?)(?=^<|\Z)'
 m=re.search(pat,text); assert m,section
 body=m[2];line=key+' = '+str(value)
 if re.search(r'(?m)^'+re.escape(key)+r'\s*=',body):body=re.sub(r'(?m)^'+re.escape(key)+r'\s*=.*$',line,body)
 else:body=body.rstrip()+'\n'+line+'\n\n'
 return text[:m.start()]+m[1]+body+text[m.end():]
settings={'mesh':{'nx1':64,'nx2':64,'nx3':64,'x1min':-32,'x1max':32,'x2min':-32,'x2max':32,'x3min':-32,'x3max':32},'meshblock':{'nx1':16,'nx2':16,'nx3':16},'mesh_refinement':{'refinement':'static','num_levels':4,'max_nmb_per_rank':128},'time':{'cfl_number':.2,'tlim':1000,'nlim':-1,'ndiag':100},'coord':{'minkowski':'false'},'z4c':{'history_excise_ks_radius':1,'debug_balance_horizon':1,'debug_snapshot_operations':'none','damp_kappa1':0,'shift_eta':.02,'residual_lapse_damping':.01},'problem':{'bh_mass':1,'bh_spin':0,'bh_background':'schwarzschild_trumpet','pure_background':'true','zero_tmunu':'true','outer_sponge_enabled':'true','outer_sponge_geometry':'radial','outer_sponge_start_radius':8,'outer_sponge_ramp_width':20,'outer_sponge_damping_time':20,'vacuum_gauge_pulse_amplitude':1e-8,'vacuum_gauge_pulse_component':0,'vacuum_gauge_pulse_x1':2.5,'vacuum_gauge_pulse_x2':0,'vacuum_gauge_pulse_x3':0,'vacuum_gauge_pulse_width':.75,'outer_sponge_test_theta_pulse_amplitude':0,'characteristic_test_family':'none','characteristic_test_amplitude':0,'force_minkowski_metric':'false','amr_bh_refine_radius':0,'amr_bh_derefine_radius':0,'amr_bh_exclusion_radius':0,'amr_bh_refine_level':-1,'amr_star_refine':'false'},'output1':{'dt':1},'output2':{'dt':1000,'single_file_per_rank':'true'}}
for sec,items in settings.items():
 for key,val in items.items():s=change(s,sec,key,val)
# No dynamic-AMR criterion is active. The explicit static cubes define all refinement.
s=re.sub(r'(?ms)^<amr_criterion0>\n.*?(?=^<|\Z)','',s)
for level,r in [(1,16),(2,8),(3,4)]:
 s+=f'\n<refined_region{level}>\nlevel = {level}\n'+''.join(f'x{i}min = {-r}\nx{i}max = {r}\n'for i in (1,2,3))
configs={
 'pulse_sponge':{},
 'zero_sponge':{'problem/vacuum_gauge_pulse_amplitude':0,'time/nlim':3},
 'pulse_nosponge':{'problem/outer_sponge_enabled':'false'},
 'zero_nosponge':{'problem/outer_sponge_enabled':'false','problem/vacuum_gauge_pulse_amplitude':0,'time/nlim':3},
 'pulse_kappa01_sponge':{'z4c/damp_kappa1':.1},
 'pulse_kappa01_nosponge':{'z4c/damp_kappa1':.1,'problem/outer_sponge_enabled':'false'},
 'pulse_original_sources_nosponge':{'z4c/damp_kappa1':.1,'z4c/shift_eta':2,'z4c/residual_lapse_damping':.1,'problem/outer_sponge_enabled':'false'},
 'pulse_sponge_halfdt':{'time/cfl_number':.1},
 'mesh32_pulse_sponge':{'meshblock/nx1':32,'meshblock/nx2':32,'meshblock/nx3':32},
 'mesh32_inset_pulse_sponge':{'meshblock/nx1':32,'meshblock/nx2':32,'meshblock/nx3':32,**{f'refined_region{level}/x{i}{side}':sign*radius for level,radius in [(1,15.99),(2,7.99),(3,3.99)]for i in (1,2,3)for side,sign in [('min',-1),('max',1)]}},
}
inputs=HERE/'inputs';inputs.mkdir(exist_ok=True);manifest={}
for name,overrides in configs.items():
 text=s
 for key,val in overrides.items():section,param=key.split('/');text=change(text,section,param,val)
 text='# Prepared only: no evolution or submission authorized by this file.\n'+text
 path=inputs/(name+'.athinput');path.write_text(text)
 manifest[name]={'path':str(path.relative_to(HERE)),'sha256':hashlib.sha256(text.encode()).hexdigest(),'overrides_from_pulse_sponge':overrides}
(HERE/'input-manifest.json').write_text(json.dumps({'base_path':str(BASE),'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),'inputs':manifest},indent=2)+'\n')
