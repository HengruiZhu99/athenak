from pathlib import Path
import json,hashlib,shutil,subprocess,difflib
w=Path(__file__).resolve().parent; root=Path.cwd()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
src=root/'build-layer-research/boundary/full-tensor-global-final/sources/full22-v2'
evo=root/'build-layer-research/time-projection-controls/composed-derivative-build/include/z4c/hyperboloidal'
diag=root/'build-layer-research/time-projection-controls/composed-diagnostic-attribution/composed-diagnose-include/z4c/hyperboloidal/cartesian_patch.hpp'
assert sha(evo/'athenak_bridge.hpp')=='06a790ee1cfb275d6c6db77507b2b7b2b719cdf79f3737cb934ea19c09e0d34d'
assert sha(evo/'cartesian_patch.hpp')=='e0c3c657723e1aa6cf1fb6015fb14f8b31514a3462404a3448b80a9fe04000b8'
e=(evo/'cartesian_patch.hpp').read_text();d=diag.read_text()
assert d==e.replace('auto numerical = LoadMeshJet<3>(fields, idx, 0, k, j, i);','auto numerical = LoadComposedMeshJet<3>(fields, idx, 0, k, j, i);',1)
inputs=w/'inputs';inputs.mkdir(exist_ok=True); inc=w/'include/z4c/hyperboloidal';inc.mkdir(parents=True,exist_ok=True)
records={}
def copy(p,q):
 records[str(p.relative_to(root))]={'sha256':sha(p),'bytes':p.stat().st_size};shutil.copyfile(p,q)
for name in ['full22_server.cpp','projected_base.hpp','old-jv-source.cpp','native_injection.hpp','spatial_norm_control.hpp','build-spatialnorm.json','diagnostic_constraint_norms.cpp','krylov_propagate.py']:
 copy(src/name,inputs/name);shutil.copyfile(inputs/name,w/name)
copy(evo/'athenak_bridge.hpp',inc/'athenak_bridge.hpp');copy(diag,inc/'cartesian_patch.hpp')
copy(evo/'cartesian_patch.hpp',inputs/'evolution-cartesian_patch.hpp')
copy(root/'build-layer-research/boundary/full-tensor-propagator/full22-v2/validate22.py',inputs/'validate22.py');shutil.copyfile(inputs/'validate22.py',w/'validate22.py')
copy(root/'build-layer-research/boundary/full-tensor-propagator/spatialnorm-validation-vectors.npz',w/'spatialnorm-validation-vectors.npz')
copy(root/'build-layer-research/boundary/full-tensor-C0-N20-20261009/full22/assemble_projected.py',inputs/'assemble_projected.py');shutil.copyfile(inputs/'assemble_projected.py',w/'assemble_projected.py')
p=w/'projected_base.hpp';s=p.read_text().replace('g.n[d]=n+6;','g.n[d]=n+8;').replace('-.5*(n+5)*g.h[d]','-.5*(n+7)*g.h[d]')
s=s.replace('void Constraints(const double*v,double*out,double scale){','void Constraints(const double*v,double*out,double scale,bool composed=true){')
s=s.replace('auto d=hyp::LoadMeshJet<3>(dev,idx,0,c.k,c.j,c.i);double vals[2][7];','auto d=composed?hyp::LoadComposedMeshJet<3>(dev,idx,0,c.k,c.j,c.i):hyp::LoadMeshJet<3>(dev,idx,0,c.k,c.j,c.i);double vals[2][7];')
s=s.replace('auto jet=hyp::LoadMeshJet<3>(dev,idx,0,c.k,c.j,c.i);','auto jet=hyp::LoadComposedMeshJet<3>(dev,idx,0,c.k,c.j,c.i);')
p.write_text('// Fresh ng4 composed-RHS/matching-H control; original M/Z/Theta and all lower-order terms retained.\n'+s)
p=w/'full22_server.cpp';s=p.read_text().replace('auto jet=hyp::LoadMeshJet<3>(dev,idx,0,c.k,c.j,c.i);','auto jet=hyp::LoadComposedMeshJet<3>(dev,idx,0,c.k,c.j,c.i);').replace('for(int b=-3;b<=3;++b)raw.insert','for(int b=-4;b<=4;++b)raw.insert').replace('d==e?Dxx<3>','d==e?hyp::ComposedDxx<3>')
s=s.replace("mode=='d'||mode=='e'","mode=='d'||mode=='c'||mode=='e'").replace("mode=='d'?7*a.cells.size()","(mode=='d'||mode=='c')?7*a.cells.size()")
s=s.replace("else if(mode=='e')a.Norms", "else if(mode=='c')a.Constraints(v.data(),out.data(),pars[0],false);else if(mode=='e')a.Norms")
p.write_text('// Fresh native radius-four raw Cartesian coefficients are substituted through strict donors exactly once.\n'+s)
p=w/'validate22.py';s=p.read_text().replace("w.parent/f'{g}-validation-vectors.npz'","w/f'{g}-validation-vectors.npz'");p.write_text(s)
p=w/'krylov_propagate.py';s=p.read_text().replace("w.parent/f'{g}-validation-vectors.npz'","w/f'{g}-validation-vectors.npz'");p.write_text(s)
p=w/'assemble_projected.py';s=p.read_text().replace(" and ref['strict_interior_donor_references']==88992", " and ref['strict_interior_donor_references']>0").replace("original C0 spatialnorm N20 finiteΩ strict interior","fresh composed ng4 C0 spatialnorm N16 finiteΩ strict interior");p.write_text(s)
(w/'input-sources.json').write_text(json.dumps({'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'sources':records,'overlay_delta':list(difflib.unified_diff(e.splitlines(),d.splitlines(),fromfile='composed-evolution',tofile='composed-matching-H'))},indent=2)+'\n')
flags=json.loads((inputs/'build-spatialnorm.json').read_text());flags.insert(3,'-I'+str(w/'include'))
for source,output in [('full22_server.cpp','server-spatialnorm'),('diagnostic_constraint_norms.cpp','diagnostic-composed')]:
 cmd=[str(w/source) if x.endswith('/full22_server.cpp') else str(w/output) if x.endswith('/server-spatialnorm') else x for x in flags]
 (w/('build-'+output+'.json')).write_text(json.dumps(cmd,indent=2)+'\n')
(w/'source-diffs.patch').write_text(''.join(''.join(difflib.unified_diff((inputs/n).read_text().splitlines(True),(w/n).read_text().splitlines(True),fromfile='frozen/'+n,tofile='fresh/'+n)) for n in ['projected_base.hpp','full22_server.cpp','validate22.py','krylov_propagate.py','assemble_projected.py']))
print('prepared',w,'sources',len(records))
