from pathlib import Path
import hashlib,json,math,subprocess,sys,time
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
mode=sys.argv[1];assert mode in ('release','debug')
R=json.loads((P/'local-recipe.json').read_text());T=R['thresholds']
for s in R['sources']:assert sha(P/s['path'])==s['sha256']
for s in R['reference_input_pins']:assert sha(P/s['path'])==s['sha256']
# The original recipe stored a bad relative spelling for this correct hash.
# Correct path is explicitly pinned in local-gate-admission.json; neither recipe nor compiled source is rewritten.
I=ROOT/'build-layer-research/continuum/immutable-independent-higher-reference-jets-20261009/index.json'
assert sha(I)=='7c556fcf4bc82671ebf0a8de7751a316363d541b7dd77e0e9a7ac86f7e94f8eb'
B=P/'build-attempts'/f'{mode}-001';build=json.loads((B/'receipt.json').read_text());exe=Path(build['executable_path']);assert sha(exe)==build['executable_sha256']
A=P/f'local-gates-{mode}-001';A.mkdir();start=time.monotonic();receipt={'kind':'finite-Omega coordinate/source local gates','mode':mode,'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'recipe_sha256':sha(P/'local-recipe.json'),'admission_sha256':sha(P/'local-gate-admission.json'),'driver_sha256':sha(__file__),'build_receipt_sha256':sha(B/'receipt.json'),'executable_sha256':sha(exe),'runs':[],'passed':False}
def finite(x):
 if isinstance(x,(int,float)):return math.isfinite(x)
 if isinstance(x,dict):return all(finite(v)for v in x.values())
 if isinstance(x,list):return all(finite(v)for v in x)
 return True
def run(name,args,data=b''):
 st=time.monotonic();cmd=[str(exe)]+args;q=subprocess.run(cmd,input=data,capture_output=True);(A/(name+'.stdout')).write_bytes(q.stdout);(A/(name+'.stderr')).write_bytes(q.stderr)
 rec={'name':name,'command':cmd,'input_sha256':hashlib.sha256(data).hexdigest(),'exit_code':q.returncode,'seconds':time.monotonic()-st,'stdout_sha256':sha(A/(name+'.stdout')),'stderr_bytes':len(q.stderr)};receipt['runs'].append(rec);assert q.returncode==0 and not q.stderr,rec
 rows=[json.loads(s)for s in q.stdout.decode().splitlines()];assert all(finite(s)for s in rows);return rows
def scaled(a,b):return max(abs(x-y)/max(1.,abs(y))for x,y in zip(a,b))
def maxabs(a):return max(abs(x)for x in a)
def fdclass(a):
 if a[-1]>T['directional_fd_final_scaled']:return 'fail_final'
 if a[-1]<=.5*a[0]:return 'truncation_decrease'
 if max(a)<=T['directional_fd_final_scaled']:return 'within_absolute_floor'
 return 'unclassified'
try:
 syn=run('synthetic',['--synthetic'])[0];assert max(syn.values())<=T['synthetic_scaled'];receipt['synthetic']=syn
 coords=[]
 for r in [.1,.5,.84,.96]:
  for d in [[.36,-.48,.8],[.8,.36,-.48]]:coords.append([r*a for a in d])
 data=''.join(' '.join(format(a,'.17g')for a in x)+'\n'for x in coords).encode();(A/'reference-input.txt').write_bytes(data);refs=run('reference-cartesian',['--reference'],data);assert len(refs)==8;receipt['cartesian_reference_export_only']={'rows':len(refs),'independent_CPP_composition_readback_pending':True}
 cases=json.loads((P/'cases.json').read_text());rows=run('coordinate',['--batch'],(P/'cases.txt').read_bytes());assert len(rows)==len(cases)
 maxima={k:0. for k in ['geometry_entry_scaled','geometry_absolute','gauge_entry_scaled','gauge_absolute','physical_constraint_scaled','physical_constraint_absolute','input_normal_scaled','input_normal_absolute','output_normal_scaled','output_normal_absolute','spatial_normal_scaled','spatial_normal_absolute','reference_rhs_absolute','reference_constraint_absolute','core_full_jet_scaled','core_acceleration_scaled','physical_vs_factored_moderate_r_scaled','fd_final_scaled']};classifications={};worst={}
 for idx,(c,a)in enumerate(zip(cases,rows)):
  vals={'geometry_entry_scaled':scaled(a['actual_rhs22'],a['kinematic_geometry22']),'geometry_absolute':maxabs([x-y for x,y in zip(a['actual_rhs22'],a['kinematic_geometry22'])]),'gauge_entry_scaled':scaled(a['actual_gauge4'],a['gauge_attribution4']),'gauge_absolute':maxabs([x-y for x,y in zip(a['actual_gauge4'],a['gauge_attribution4'])]),'physical_constraint_absolute':maxabs(a['physical_constraints8']),'input_normal_absolute':maxabs(a['input_normals']),'output_normal_absolute':maxabs(a['output_normals']),'spatial_normal_absolute':a['spatial_normal_max'],'reference_rhs_absolute':a['reference_rhs_max'],'reference_constraint_absolute':maxabs(a['reference_constraints8']),'core_full_jet_scaled':a['core_lift_error'],'core_acceleration_scaled':scaled(a['acceleration4'],a['core_acceleration4'])if c['r']<=.05 else 0.,'physical_vs_factored_moderate_r_scaled':a['physical_vs_factored_lift_scaled']if c['r']<=.95 else 0.,'fd_final_scaled':a['double_fd_errors'][-1]}
  vals.update(physical_constraint_scaled=vals['physical_constraint_absolute']/a['input_jet_scale'],input_normal_scaled=vals['input_normal_absolute']/max(1.,a['input_value_norm']),output_normal_scaled=vals['output_normal_absolute']/max(1.,a['rhs_norm']),spatial_normal_scaled=vals['spatial_normal_absolute']/a['input_jet_scale'])
  for k,v in vals.items():
   if v>maxima[k]:maxima[k]=v;worst[k]={'case':idx,'r':c['r'],'name':c['name'],'point':c['point'],'value':v}
  cl=fdclass(a['double_fd_errors']);classifications[cl]=classifications.get(cl,0)+1
 checks={'physical_vs_factored_identity_moderate_r':maxima['physical_vs_factored_moderate_r_scaled']<=T['physical_vs_factored_moderate_r_scaled'],'geometry':maxima['geometry_entry_scaled']<=T['geometry_rhs_entry_scaled'],'gauge':maxima['gauge_entry_scaled']<=T['gauge_attribution_entry_scaled'],'physical_constraint':maxima['physical_constraint_scaled']<=T['physical_constraint_scaled_by_full_input_jet'],'input_normal':maxima['input_normal_scaled']<=T['input_normal_scaled_by_input_values'],'output_normal':maxima['output_normal_scaled']<=T['output_normal_scaled_by_rhs'],'spatial_normal':maxima['spatial_normal_scaled']<=T['spatial_normal_scaled_by_full_input_jet'],'reference_rhs':maxima['reference_rhs_absolute']<=T['reference_rhs_absolute'],'reference_constraint':maxima['reference_constraint_absolute']<=T['reference_physical_constraint_absolute'],'core_lift':maxima['core_full_jet_scaled']<=T['core_full_jet_scaled'],'core_acceleration':maxima['core_acceleration_scaled']<=T['core_acceleration_scaled'],'coordinate_directional_FD':not any(k in classifications for k in ['fail_final','unclassified'])}
 receipt.update(coordinate_cases=len(rows),coordinate_maxima=maxima,coordinate_worst=worst,coordinate_checks=checks,coordinate_fd_classifications=classifications)
 binding=run('binding',['--binding']);assert len(binding)==421;negative=binding[-1]['negative_double_only_wrapper_difference'];bclass={}
 for row in binding[:-1]:cl=fdclass(row['fd_errors']);bclass[cl]=bclass.get(cl,0)+1
 receipt.update(binding_cases=420,binding_fd_final_max=max(r['fd_errors'][-1]for r in binding[:-1]),binding_fd_classifications=bclass,negative_wrapper_difference=negative)
 checks['raw22_and_chart20_directional_FD']=not any(k in bclass for k in ['fail_final','unclassified']);checks['negative_double_only_wrapper']=negative>=T['negative_wrapper_min_difference'];receipt['passed']=all(checks.values())
except Exception as e:receipt['exception']=repr(e)
receipt['seconds']=time.monotonic()-start;receipt['executable_unchanged']=sha(exe)==build['executable_sha256'];receipt['sources_unchanged']=all(sha(P/s['path'])==s['sha256']for s in R['sources']);(A/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps({'path':str(A),'receipt_sha256':sha(A/'receipt.json'),**receipt},indent=2));sys.exit(0 if receipt['passed']else 1)
