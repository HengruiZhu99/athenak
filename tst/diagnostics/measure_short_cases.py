"""Small, reproducible CPU overhead and metric interpolation-resolution measurements."""
import argparse,json,pathlib,re,statistics,subprocess,time
import sys,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'vis/python'));import athena_read
p=argparse.ArgumentParser();p.add_argument('executable',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True);exe=str(a.executable.resolve())
base=(ROOT/'tst/diagnostics/smoke.athinput').read_text()
results={'backend':'Kokkos Serial, MPI build, one rank','overhead':{},'area_resolution':[]}
def run(name,text):
 d=a.output/name;d.mkdir();inp=d/'input';inp.write_text(text)
 start=time.perf_counter();r=subprocess.run([exe,'-i',str(inp),'-d',str(d)],capture_output=True,text=True)
 (d/'run.log').write_text(r.stdout+r.stderr)
 if r.returncode:raise RuntimeError(r.stdout+r.stderr)
 elapsed=time.perf_counter()-start
 cpu=float(re.search(r'cpu time used\s*=\s*([\deE+.-]+)',r.stdout)[1])
 return elapsed,cpu,d
for mode in ('disabled','surfaces','budgets'):
 records=[]
 for repeat in range(3):
  text=base.replace('nx1 = 16','nx1 = 32').replace('nx2 = 16','nx2 = 32').replace('nx3 = 16','nx3 = 32')
  text=text.replace('nlim = 3','nlim = 20').replace('tlim = 1\n','tlim = 100\n')
  text=text.replace('flux_ntheta = 8','flux_ntheta = 16').replace('flux_nphi = 16','flux_nphi = 32')
  text=text.replace('dcycle = 1','dcycle = 5').replace('dcycle = 2','dcycle = 100')
  if mode=='disabled':text=text.replace('flux_diagnostics = true','flux_diagnostics = false')
  if mode=='budgets':text=text.replace('<problem>','<problem>\nflux_budget=true\nbudget_r_inner=3.5\nbudget_r_outer=4\nbudget_subcells=2')
  wall,cpu,_=run(f'{mode}-{repeat}',text);records.append({'wall_seconds':wall,'evolution_seconds':cpu})
 results['overhead'][mode]={'runs':records,'median_evolution_seconds':statistics.median(x['evolution_seconds'] for x in records)}
for count in (16,32,64):
 text=base.replace('nx1 = 16',f'nx1 = {count}').replace('nx2 = 16',f'nx2 = {count}').replace('nx3 = 16',f'nx3 = {count}')
 text=text.replace('nlim = 3','nlim = 0').replace('flux_ntheta = 8','flux_ntheta = 64').replace('flux_nphi = 16','flux_nphi = 128')
 _,_,d=run(f'area-{count}',text)
 h=athena_read.hst(d/'matched_bondi.user.hst');area=h['area_BH1_Kerr'][-1]
 results['area_resolution'].append({'cells_per_axis':count,'dx':16/count,'proper_area':float(area),'relative_error':float(area/(16*np.pi)-1),'valid':bool(h['valid_BH1_Kerr'][-1])})
(a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
