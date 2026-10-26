#!/usr/bin/env python3
"""Frozen-source, bounded coupled refinement controls; no stability inferred from timeout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
from run_conformal_spherical import run_case, sha
from run_spherical_puncture import parameters


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary',type=Path);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--phase',choices=['puncture','pulse'],default='puncture')
    p.add_argument('--timeout',type=int,default=420)
    p.add_argument('--tlim',type=float,default=.02)
    p.add_argument('--grids',type=Path,required=True)
    a=p.parse_args();assert 0<a.timeout<=450
    source=a.source.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'athena_frozen';shutil.copy2(a.binary.resolve(),binary)
    grids=json.loads(a.grids.read_text());assert 1<=len(grids)<=6
    assert len({g['name'] for g in grids})==len(grids)
    files=[*source.glob('src/**/*.cpp'),*source.glob('src/**/*.hpp'),source/'CMakeLists.txt',source/'src/CMakeLists.txt',
           source/'inputs/z4c/cubed_sphere_conformal_probe.athinput',Path(__file__).resolve(),
           Path(__file__).with_name('run_conformal_spherical.py'),Path(__file__).with_name('run_spherical_puncture.py')]
    launch={'phase':a.phase,'binary_sha256':sha(binary),'grids':grids,
            'source_sha256':{str(f.relative_to(source)):sha(f) for f in sorted(set(files)) if not f.name.startswith('._')},
            'scope':'One rank, uniform Cartesian bulk, original equations, immutable Minkowski reference; coupled grid and temporal controls. Completion alone does not establish scientific accuracy.'}
    shutil.copy2(a.grids,out/'grids.json');(out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    def run(item):
        index,g=item;extra=parameters() if a.phase=='puncture' else {}
        extra.update(g.get('extra',{}))
        if g.get('checkpoint'):
            extra['cubed_sphere/conformal_checkpoint']=str(out/g['name']/'probe_checkpoint.bin')
        if g.get('restart'):
            assert sha(Path(g['restart']))==g['restart_sha256'],'restart payload changed'
            extra['cubed_sphere/conformal_restart']=g['restart']
        if g.get('checkpoint') or g.get('restart'):
            extra['cubed_sphere/conformal_checkpoint_tag']=sha(binary)
        extra['cubed_sphere/conformal_fixed_dt']=g.get('dt',.00008)
        return run_case(binary,source,out,g['name'],g.get('device',f'{index}.0'),g['bulk'],g['nr'],g['na'],
                        g.get('amplitude',0 if a.phase=='puncture' else .2),g.get('kappa',5 if a.phase=='puncture' else 10),
                        g.get('cfl',.15 if a.phase=='puncture' else .1),g.get('tlim',a.tlim),g.get('degree',3 if a.phase=='puncture' else 2),
                        a.timeout,g.get('pole_cfl',.04 if a.phase=='puncture' else .03),extra)
    with ThreadPoolExecutor(max_workers=len(grids)) as pool:
        results=list(pool.map(run,enumerate(grids)))
    launch['cases']={g['name']:r for g,r in zip(grids,results)}
    (out/'summary.json').write_text(json.dumps(launch,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
