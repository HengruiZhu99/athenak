"""Source-only exact native overlay/commands. This program never invokes a compiler."""
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'build-layer-spatial-norm-native'
FAMILY=ROOT/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'
HELPER=ROOT/'build-layer-research/continuum/reference-wave-map-gauge-20261009/immutable-local-reference-wave-map-20261009/reference_wave_map.hpp'
PINS={
 'src/z4c/hyperboloidal/cartesian_patch.hpp':'fecbebdbb8f82cf5aef72e2bf083a0e25314978a840f103771d1530c1f369280',
 'src/z4c/z4c_newdt.cpp':'0aebb2b7d88907506f2dd681008b1237a2e7382368f8e3c239bfc606511950bd',
 'build-layer-spatial-norm-native/src/athena':'dd1d189210abd4e094da339dd73e3014357924b343c9f08f658eaf8cd4ae172d',
 'build-layer-spatial-norm-native/compile_commands.json':'d82a531670a87cb8d9b3bdf56e795ec1e25075612b3d353c424b4db3bdfb7e8a',
 'build-layer-spatial-norm-native/src/CMakeFiles/athena.dir/link.txt':'7cf6f8ea8c914e7439d03cfa0d8941f1a8930d137c3b3e9b09f1c66e16130033',
 'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family/native-build-receipt.json':'a39430b8c3bcf2e7cc05d03a441831edc493f050afbb687e642418f1e5f56ce6',
 str(HELPER.relative_to(ROOT)):'56d61c56bf37bf33591bec74c029c686c7d39f7abbe62dd090becda8e7176e28'}
IMPLEMENTATION='27c19d20696ea6dd4704032c51dfd026218f64f2'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def once(s,a,b):
    assert s.count(a)==1, ('nonunique exact source seam',a,s.count(a))
    return s.replace(a,b)

def main():
    assert sys.argv[1:] in ([],['--prepare-only'])
    for name,digest in PINS.items(): assert sha(ROOT/name)==digest,name
    base_receipt=json.loads((FAMILY/'native-build-receipt.json').read_text())
    assert base_receipt['implementation_commit']==IMPLEMENTATION
    for name,digest in base_receipt['source_sha256'].items():
        assert sha(ROOT/name)==digest,name
    for name,digest in base_receipt['overlay_sha256'].items():
        assert sha(FAMILY/name)==digest,name
    cart_path=ROOT/'src/z4c/hyperboloidal/cartesian_patch.hpp'
    public=cart_path.read_text()
    wave=once(public,'#include "z4c/hyperboloidal/layer_gauge.hpp"',
                     '#include "z4c/hyperboloidal/layer_gauge.hpp"\n#include "native_wave_map.hpp"')
    seam='''      const Real idx[3] = {1/spacing[0],1/spacing[1],1/spacing[2]};
      const auto p = ref.At(g.first[0]+i*g.h[0],g.first[1]+j*g.h[1],g.first[2]+k*g.h[2]);'''
    xyz='''      const Real idx[3] = {1/spacing[0],1/spacing[1],1/spacing[2]};
      const Real xyz[3] = {g.first[0]+i*g.h[0],g.first[1]+j*g.h[1],g.first[2]+k*g.h[2]};
      const auto p = ref.At(xyz[0],xyz[1],xyz[2]);'''
    wave=once(wave,seam,xyz)
    gauge='''          || !AssembleGaugeInterior((ref.layer.enabled || lg.physical_trace_lapse)
              ? InteriorLayerGauge(p,u,lg)
              : UnfactoredReferenceGauge(ref,p,u,gauge,true,true),
                                     omega.omega,gauge_rhs)) {'''
    replaced='''          || !ResearchNativeWaveMapGauge(p,u,xyz,gauge_rhs)) {'''
    wave=once(wave,gauge,replaced)
    # Exact inverse patch proves that no other CartesianPatch source changed.
    restored=wave.replace(replaced,gauge).replace(xyz,seam).replace(
        '#include "z4c/hyperboloidal/layer_gauge.hpp"\n#include "native_wave_map.hpp"',
        '#include "z4c/hyperboloidal/layer_gauge.hpp"')
    assert restored.encode()==cart_path.read_bytes()
    public_dt=(ROOT/'src/z4c/z4c_newdt.cpp').read_text()
    dt_seam='''        /pmy_pack->pmesh->cfl_no;
    return TaskStatus::complete;'''
    half_dt=once(public_dt,dt_seam,'''        /pmy_pack->pmesh->cfl_no;
    // Private timestep-only control: halve BOTH admitted native caps.
    dtnew *= Real(0.5);
    return TaskStatus::complete;''')
    commands=json.loads((BASE/'compile_commands.json').read_text())
    selected=[]
    for e in commands:
        # Kokkos generated objects need not have native depfiles; select only src.
        if not Path(e['file']).is_relative_to(ROOT/'src'): continue
        cmd=shlex.split(e['command']); assert '-o' in cmd
        obj=(Path(e['directory'])/cmd[cmd.index('-o')+1]).resolve()
        deps=Path(str(obj)+'.d'); assert deps.is_file(),deps
        if str(cart_path) in deps.read_text(): selected.append((e,cmd,obj,deps))
    assert len(selected)==6,len(selected)
    assert any(e['file'].endswith('/z4c_newdt.cpp') for e,_,_,_ in selected)
    link=shlex.split((BASE/'src/CMakeFiles/athena.dir/link.txt').read_text())
    link_cwd=BASE/'src'
    linked={str((link_cwd/a).resolve()):{'sha256':sha((link_cwd/a).resolve()),
             'bytes':(link_cwd/a).resolve().stat().st_size}
            for a in link if a.endswith(('.o','.a'))}
    assert len([x for x in linked if x.endswith('.o')])==182
    recipes=HERE/'recipes'; recipes.mkdir(exist_ok=False)
    for mode,half in [('wave-map',False),('c0',False),('wave-map-half',True)]:
        out=recipes/mode; inc=out/'include'; overlay=inc/'z4c/hyperboloidal'
        overlay.mkdir(parents=True)
        cart=overlay/'cartesian_patch.hpp'; cart.write_text(public if mode=='c0' else wave)
        helper=inc/'reference_wave_map.hpp'; helper.write_bytes(HELPER.read_bytes())
        injection=inc/'native_wave_map.hpp'
        injection.write_bytes((HERE/'include/native_wave_map.hpp').read_bytes())
        private_sources=[cart,helper,injection]
        dt_path=None
        if half:
            dt_path=out/'z4c_newdt.cpp'; dt_path.write_text(half_dt); private_sources.append(dt_path)
        patch=out/'cartesian-overlay.patch'
        patch.write_text(''.join(difflib.unified_diff(public.splitlines(True),
            cart.read_text().splitlines(True),fromfile='public/cartesian_patch.hpp',
            tofile=mode+'/cartesian_patch.hpp')))
        if half:
            (out/'timestep-only.patch').write_text(''.join(difflib.unified_diff(
                public_dt.splitlines(True),half_dt.splitlines(True),
                fromfile='public/z4c_newdt.cpp',tofile='wave-map-half/z4c_newdt.cpp')))
        compile_rows=[]; private_link=list(link)
        for k,(e,old,obj,dep) in enumerate(selected):
            cmd=list(old)
            forced=cmd.index('-include')
            assert cmd.count('-include')==1
            assert Path(cmd[forced+1]).resolve()==FAMILY/'native_injection.hpp'
            del cmd[forced:forced+2]  # remove old spatial-norm macros completely
            cmd[1:1]=['-I'+str(inc)]
            output=out/('object-'+str(k)+'.o'); depout=out/('object-'+str(k)+'.d')
            cmd[cmd.index('-o')+1]=str(output)
            if half and e['file'].endswith('/z4c_newdt.cpp'):
                cmd[cmd.index(e['file'])]=str(dt_path)
                # Local includes in this copied .cpp need the original source dir.
                cmd[1:1]=['-I'+str(ROOT/'src/z4c')]
            cmd.extend(['-MD','-MF',str(depout)])
            assert '-include' not in cmd
            compile_rows.append({'command':cmd,'cwd':e['directory'],
                'original_source':e['file'],'base_object':str(obj),
                'base_object_sha256':sha(obj),'base_depfile_sha256':sha(dep),
                'output':str(output),'depfile':str(depout)})
            matches=[j for j,a in enumerate(private_link)
                     if a.endswith('.o') and (link_cwd/a).resolve()==obj]
            assert len(matches)==1
            private_link[matches[0]]=str(output)
        exe=out/('athena-'+mode)
        private_link[private_link.index('-o')+1]=str(exe)
        record={'scope':'HELD source-only actual native build recipe; no compilation/evolution',
          'mode':mode,'half_step':half,'compiled_implementation':IMPLEMENTATION,
          'source_preparation_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'scientific_prerequisites':json.loads((HERE/'prerequisites.json').read_text()),
          'prepare_script_sha256':sha(Path(__file__)),
          'base_inputs_sha256':PINS,'base_source_sha256':base_receipt['source_sha256'],
          'base_overlay_sha256':base_receipt['overlay_sha256'],
          'base_link_inputs':linked,
          'private_sources_sha256':{str(p.relative_to(ROOT)):sha(p) for p in private_sources},
          'exact_inverse_cartesian_patch_restores_public':True,
          'old_spatialnorm_forced_include_removed':True,
          'compile_commands':compile_rows,'link_command':private_link,
          'link_cwd':str(link_cwd),'executable':str(exe),
          'compilation_executed':False,'native_execution_authorized':False}
        dump(out/'recipe.json',record)
    dump(HERE/'source-preparation-receipt.json',{
      'passed_source_preparation':True,'no_compiler_or_native_process_called':True,
      'six_source_files':[e['file'] for e,_,_,_ in selected],
      'recipes':{str((recipes/m/'recipe.json').relative_to(HERE)):sha(recipes/m/'recipe.json')
                 for m in ['wave-map','c0','wave-map-half']},
      'production_source_unchanged':True,'base_executable_unchanged':True})
    print('PASS source-only recipes: wave-map, matched C0, exact half-cap wave-map; six TUs each')

if __name__=='__main__': main()
