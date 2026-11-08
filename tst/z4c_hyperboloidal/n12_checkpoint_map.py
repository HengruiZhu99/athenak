#!/usr/bin/env python3
"""Build the independent fixed-N12 coordinate/order proof and portable sidecar."""
import argparse
import json
from pathlib import Path
import re
import shlex
import struct
import subprocess

import numpy as np
from n12_checkpoint import (DT, HEADER, NBULK, NSURFACE, parse_native, require,
                            sha256, validate_header)

REQUIRED = {
    'mesh/nghost': 4, 'mesh/nx1': 36, 'mesh/nx2': 36, 'mesh/nx3': 36,
    'mesh/x1min': -.1, 'mesh/x1max': .1, 'mesh/x2min': -.1, 'mesh/x2max': .1,
    'mesh/x3min': -.1, 'mesh/x3max': .1,
    'meshblock/nx1': 12, 'meshblock/nx2': 12, 'meshblock/nx3': 12,
    'mesh/logical_location_version': 2, 'mesh_atlas/type': 'cubed_sphere',
    'mesh_atlas/spherical_inner_radius': .08, 'mesh_atlas/evolved_overlap': 3,
    'mesh_atlas/outer_endpoint_grid': 'true', 'mesh_refinement/refinement': 'static',
    'time/cfl_number': .2, 'time/evolution': 'dynamic',
    'time/integrator': 'rk4_classical', 'time/dt': DT,
    'problem/pgen_name': 'z4c_hyperboloidal_puncture',
    'z4c_hyperboloidal/inner_z4c_radius_M': 20,
    'z4c_hyperboloidal/compact_slicing_end': .97,
    'z4c_hyperboloidal/outer_lapse_damping': .25,
    'z4c_hyperboloidal/ko': .1, 'z4c_hyperboloidal/chi_floor': 1e-10,
    'z4c_hyperboloidal/scri_penalty': .5,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--template-checkpoint', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    build, legacy, output = args.build.resolve(), args.legacy.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    template = parse_native(args.template_checkpoint)
    require(template['cycle'] == 0, 'mapping template must be a native initial checkpoint')
    files = [p for p in (root/'src').rglob('*') if p.suffix in ('.cpp', '.hpp')]
    files += [root/'tst/z4c_hyperboloidal'/name for name in
              ('n12_checkpoint_map.cpp', 'n12_checkpoint_map.py', 'n12_checkpoint.py',
               'n12_checkpoint_contract.md', 'n12_initialize.athinput')]
    sources = {str(path): sha256(path) for path in files}
    flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
    includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M).group(1))
    options = ['BROAD_SMOOTH_A', 'CONFIGURED_FORMULATION', 'DIRECT_OUTER', 'INNER_TELEGRAPHER',
               'NORMALIZED_INNER_SHIFT', 'RESTORE_INNER_DAMPING', 'SHORT_INNER_GAUGE',
               'SMOOTH_GAUGE_BLEND', 'SPHERICAL_POLICY']
    obj, exe, dep = output/'mapping.o', output/'mapping', output/'mapping.d'
    compile_command = ['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off', *includes,
        *['-DHYPERBOLIC_'+x+'=1' for x in options], '-I'+str(legacy/'src'), '-I'+str(legacy),
        '-MMD', '-MF', str(dep), '-c', str(root/'tst/z4c_hyperboloidal/n12_checkpoint_map.cpp'),
        '-o', str(obj)]
    link = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
    link = [str(obj) if token == 'CMakeFiles/athena.dir/main.cpp.o' else token for token in link]
    require(str(obj) in link, 'native link command did not expose main object')
    link[link.index('-o')+1] = str(exe)
    objects = {str((build/'src'/token).resolve()): sha256(build/'src'/token)
               for token in link if token.endswith('.o') and token != str(obj)}
    makefile = output/'mapping.mk'
    makefile.write_text('all:\n\t'+shlex.join(compile_command)+'\n\t'+shlex.join(link)+'\n')
    with (output/'build.log').open('w') as log:
        subprocess.run(['make', '-j64', '-f', str(makefile)], cwd=build/'src',
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    dependencies = shlex.split(dep.read_text().replace('\\\n', ' ').split(':', 1)[1])
    deps = {str((build/'src'/name).resolve()): sha256((build/'src'/name).resolve())
            for name in dependencies}
    command = ['mpiexec', '-n', '1', str(exe),
               str(root/'tst/z4c_hyperboloidal/n12_initialize.athinput'), str(output)]
    with (output/'run.log').open('w') as log:
        subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (output/'mapping.bin').open('rb') as stream:
        require(stream.read(16).rstrip(b'\0') == b'N12_MAP_V1', 'unknown mapping binary')
        require(struct.unpack('<4Q', stream.read(32)) == (NBULK, NSURFACE, 106, 64),
                'mapping dimensions differ')
        records = np.fromfile(stream, dtype=np.dtype([('keys', '<i4', (6,)), ('values', '<f8', (5,))]))
    require(len(records) == NBULK+NSURFACE, 'incomplete mapping output')
    arrays = {}
    for region, section in [('bulk', records[:NBULK]), ('surface', records[NBULK:])]:
        arrays[region+'_keys'] = section['keys'].copy()
        arrays[region+'_xyz'] = section['values'][:, :3].copy()
        arrays[region+'_weight'] = section['values'][:, 3].copy()
        arrays[region+'_diagnostic_weight'] = section['values'][:, 4].copy()
    np.savez_compressed(output/'mapping.npz', **arrays)
    require((output/'topology.bin').read_bytes() == template['topology'], 'native template topology mismatch')
    require((output/'atlas.txt').read_bytes() == template['atlas'], 'native template atlas mismatch')
    manifest = dict(schema='native-n12-legacy-map-v1', mapping_sha256=sha256(output/'mapping.npz'),
        coordinate_mismatches=0, coordinate_comparison='bitwise-float64', leaves=106,
        bulk_points=NBULK, scri_points=NSURFACE, records='legacy active/row order',
        keys=['native_gid', 'restart_slot', 'i-relative-origin', 'j-relative-origin',
              'k-relative-origin', 'native-canonical-ordinal'],
        error_weights='legacy evolved-point volume including overlap and physical nodal halfweight; scri area',
        diagnostic_weights='legacy unique-coverage blend; evolved extras zero; not used to hide full-state errors',
        fixed_mesh_header_hex=template['fixed'][:-20].hex(), topology_hex=template['topology'].hex(),
        atlas_identity=template['atlas'].decode(), physics_identity=template['identity'].decode(),
        required_parameters=REQUIRED, template_checkpoint=str(template['path']),
        template_checkpoint_sha256=sha256(template['path']),
        provenance=dict(source_sha256=sources, object_sha256=objects, dependency_sha256=deps,
                        executable_sha256=sha256(exe), compile_command=compile_command, command=command))
    validate_header(template, manifest, 0)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    initial = output/'initial-reference';initial.mkdir()
    for old_name, name in [('initial-fields.bin', 'fields.bin'), ('initial-boundary.bin', 'boundary.bin')]:
        (output/old_name).rename(initial/name)
    (initial/'time.json').write_text(json.dumps(dict(steps=0, t=0., t_M=0., dt=DT))+'\n')
    (initial/'reference-manifest.json').write_text(json.dumps(dict(mapping_sha256=manifest['mapping_sha256'],
        scope='independent legacy initial reference, no native field substitution',
        sha256={name: sha256(initial/name) for name in ('fields.bin', 'boundary.bin')}), indent=2)+'\n')
    require(all(sha256(Path(name)) == digest for group in (sources, objects, deps)
                for name, digest in group.items()), 'sources/objects changed during mapping proof')
    print(json.dumps({key: manifest[key] for key in ('schema', 'mapping_sha256',
          'coordinate_mismatches', 'bulk_points', 'scri_points')}, indent=2))


if __name__ == '__main__':
    main()
