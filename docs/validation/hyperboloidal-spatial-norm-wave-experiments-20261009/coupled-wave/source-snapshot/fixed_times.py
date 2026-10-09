"""Matched-time actual RK4 audit. Reuse frozen export and exact-profile definitions."""
from pathlib import Path
import hashlib,json,time
w=Path(__file__).resolve().parent
source=w/'run_wave.py';code=source.read_text();assert code.count('receipt={')==1
# This prefix parses the same documented CLI, exports/validates the native matrix,
# and defines exact(t), energies(), A, G, q. It performs no pulse evolution.
exec(compile(code.split('receipt={')[0],str(source),'exec'),globals())
source_hash=sha(source);dt=args.dt_factor*h/meta['max_outgoing'];v,_,_=exact(0);times=[0.,.01,.025,.05,.1,.2,.5,1.,2.,4.,6.];hist=[];states=[];now=0.;steps=0
for target in times:
 while now<target:
  ds=min(dt,target-now);k1=A@v;k2=A@(v+ds*k1/2);k3=A@(v+ds*k2/2);k4=A@(v+ds*k3);v+=ds*(k1+2*k2+2*k3+k4)/6;now+=ds;steps+=1
 ex,_,eg=exact(target);grad=np.column_stack([d@v[:m] for d in G]);E,En=energies(v,grad);Ee,Ene=energies(ex,eg);e=v-ex
 hist.append({'time':target,'error_linf':float(abs(e).max()),'error_rms':float(np.sqrt(np.mean(e**2))),'phi_error_rms':float(np.sqrt(np.mean(e[:m]**2))),'pi_error_rms':float(np.sqrt(np.mean(e[m:]**2))),'killing_energy':E,'analytic_sampled_killing_energy':Ee,'positive_normal_energy':En,'both_linf':float(abs(v).max())});states.append(v.copy())
file=Path(str(prefix)+f'-fixed-dt{args.dt_factor}.npz');np.savez_compressed(file,times=times,states=np.array(states));receipt={'name':name,'native_export':meta,'command_arguments':vars(args),'dt_nominal':dt,'steps':steps,'sampled_states_sha256':sha(file),'sampled_states_path':str(file),'driver_sha256':sha(Path(__file__)),'consumed_exact_driver_sha256':source_hash,'export_source_sha256':sha(w/'export_wave.cpp'),'export_executable_sha256':sha(w/'export-wave'),'matrix_sha256':sha(Path(str(prefix)+'.mtx')),'history':hist};output=Path(str(prefix)+f'-fixed-dt{args.dt_factor}.json');output.write_text(json.dumps(receipt,indent=2)+'\n');print(output.name,hist,flush=True)
