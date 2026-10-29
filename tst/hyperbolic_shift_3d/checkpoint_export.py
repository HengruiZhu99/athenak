#!/usr/bin/env python3
"""Local, streaming export of admitted MPI snapshots for existing 3D/1D tools.

Example (no evolution is run):
  python3 checkpoint_export.py RUN.checkpoint LOCAL_EXPORT \
    --reanalyze /absolute/path/hyperbolic_shift_3d_reanalyze
  python3 compare_coupled.py --runs LOCAL_EXPORT --baseline MATCHED_1D \
    --output LOCAL_COMPARISON

The default report is RUN.report.json. --expected-time rejects a mismatched
snapshot before exporting. --worker-status plus --case checks the checkpoint
and report hashes recorded by the Aurora worker. --source-manifest additionally
checks that manifest's hash against the worker source binding; --source-root
can verify every source file when a corresponding local source copy exists.
Without execution evidence, provenance is explicitly marked unverified.

The V1/V2/V3 checkpoints are little-endian binary64 with a 72-byte header and four
contiguous arrays: volume state, volume RK compensation, scri state, scri RK
compensation. V1 is the unprojected control; V2 applies standard algebraic tensor
repair at accepted RK steps. V3 additionally exchanges angular scri stencils
across patch faces. V4 adds a 16-byte <dii configuration record after the base
header: finite nonnegative scri-dissipation epsilon, degree 7, reserved zero.
Its arrays therefore start at byte 88. V4 retains projection and cross-face
stencils even at epsilon zero. The evolution binaries reject cross-scheme
restarts and V4 rejects a different epsilon.
V5 retains that dissipation record and adds 16 bytes <4i for the normal closure:
scheme=1, radial_nodes=6, bulk_derivative_order=4, reserved=0. Its payload begins
at byte104 and its sampled-native-numerator scheme cannot mix with V1-V4.
V6 has the same byte offset but requires the exact tuple (2,7,4,3), binding
seven-node sampled normal derivatives and direct radial derivatives in the
last three volume rows. V5 and V6 checkpoints cannot be interchanged.
V6 reanalysis explicitly requests the donor_seven bulk diagnostic operator;
an older reanalysis binary rejects that extra argument. Ray diagnostics still
use independent interpolated jets and are not native cell constraints.
The header contains no topology descriptor. This exporter therefore
supports only the current unrefined 19-block layout: core half-width .125,
shell radii .45/.75/1, one core block and six faces per shell layer. All four
arrays are checked for finite values even when compensation is not exported.

Outputs are evolution.state.csv, evolution.boundary.csv, analytic fixed-mesh
evolution.coordinates.csv, evolution.json, and export.json. Use
--include-compensation to export both compensation arrays too. --reanalyze
runs only the existing local diagnostic executable and creates diagnostics/ray
CSVs. --drop-intermediates then removes only the CSV arrays made by this
invocation; the checkpoint is never modified. No jobs or 1D runs are launched.

Compatibility: evolution.json status describes completion of this snapshot
export, not completion of the source campaign. source_status, source_target_time,
and source_target_reached retain that distinction for walltime checkpoints.
Missing source diagnostics, including compatibility_max, remain null.

Momentum conventions (also retained in export.json): M_r=n^i M_i uses the
Euclidean coordinate radial direction. It is NOT an invariant norm. M_norm in
coupled_diagnostics.hpp is sqrt(gbar^{ij} M_i M_j), with gbar=gtilde/chi.
The physical norm is Omega*M_norm, Omega=(1-r^2)/2. For exactly spherical data,
M_norm=sqrt(chi/g_rr)*abs(M_r). MPI M_L2 uses M_norm; M_physical_L2 uses the
physical norm. Both use normalized compactified Cartesian coordinate volume,
not physical proper volume. compare_coupled.py plots the radial component and
reports radial and invariant norms separately. Its horizon is a 14-ray
coordinate-sphere diagnostic; native_fastflow retains the actual finder result.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

HEADER = struct.Struct('<16s4i3Q2d')
V4_CONFIG = struct.Struct('<dii')
V5_NORMAL_CONFIG = struct.Struct('<4i')
SAMPLED_NORMAL_SCHEME = 'sampled_native_numerator6'
HEADER_KEYS = ('magic', 'n', 'nf', 'kind', 'admitted', 'cells', 'surface', 'steps', 't', 'dt')
KINDS = {0: 'minkowski', 1: 'pulse', 2: 'puncture'}
CHUNK = 1 << 20
ROOT = Path(__file__).resolve().parents[2]
MOMENTUM = dict(
    radial='M_r = (x^i/r) M_i; coordinate radial component, not invariant',
    conformal='M_norm = sqrt(gbar^{ij} M_i M_j), gbar_ij = gtilde_ij/chi',
    physical='M_physical_norm = Omega*M_norm, Omega=(1-r^2)/2',
    spherical_reduction='M_norm = sqrt(chi/g_rr)*abs(M_r) only for radial momentum',
    measure='normalized compactified Cartesian coordinate volume',
    mpi_names='M_L2 is conformal invariant; M_physical_L2 is physical invariant',
    comparison_names='compare_coupled plots M_r; constraint_rms keeps M_r and M_norm separate')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f'Nonstandard JSON number {value} in {path}')
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def atomic_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def validate_checkpoint(path, expected_hash=None, expected_time=None):
    """Read bounded chunks, rejecting an unsupported layout or any nonfinite field."""
    with path.open('rb') as stream:
        raw = stream.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError('Truncated checkpoint header')
        h = dict(zip(HEADER_KEYS, HEADER.unpack(raw)))
        if h['magic'] not in [s.ljust(16,b'\0') for s in [b'HYP3D_MPI_V1',b'HYP3D_MPI_V2',b'HYP3D_MPI_V3',b'HYP3D_MPI_V4',b'HYP3D_MPI_V5',b'HYP3D_MPI_V6']] or h['nf'] != 31:
            raise ValueError('Unsupported checkpoint magic/schema/precision')
        h.update(header_bytes=HEADER.size,scri_dissipation_epsilon=0.0,scri_dissipation_degree=0)
        if h['magic'] in [s.ljust(16,b'\0') for s in [b'HYP3D_MPI_V4',b'HYP3D_MPI_V5',b'HYP3D_MPI_V6']]:
            extra = stream.read(V4_CONFIG.size)
            if len(extra) != V4_CONFIG.size:
                raise ValueError('Truncated V4 checkpoint configuration')
            epsilon,degree,reserved = V4_CONFIG.unpack(extra)
            if not math.isfinite(epsilon) or epsilon < 0 or degree != 7 or reserved != 0:
                raise ValueError('Invalid V4 checkpoint dissipation configuration')
            raw += extra
            h.update(header_bytes=HEADER.size+V4_CONFIG.size,
                     scri_dissipation_epsilon=epsilon,scri_dissipation_degree=degree)
        if h['magic'] in [b'HYP3D_MPI_V5'.ljust(16,b'\0'),b'HYP3D_MPI_V6'.ljust(16,b'\0')]:
            radial=h['magic']==b'HYP3D_MPI_V6'.ljust(16,b'\0')
            extra = stream.read(V5_NORMAL_CONFIG.size)
            if len(extra) != V5_NORMAL_CONFIG.size:
                raise ValueError('Truncated V5 checkpoint normal configuration')
            if V5_NORMAL_CONFIG.unpack(extra) != ((2,7,4,3) if radial else (1,6,4,0)):
                raise ValueError('Invalid V5 checkpoint normal derivative configuration')
            raw += extra
            h.update(header_bytes=104,scri_normal_derivative='sampled_native_numerator7' if radial else SAMPLED_NORMAL_SCHEME,
                     scri_normal_radial_nodes=7 if radial else 6,scri_normal_bulk_derivative_order=4)
            if radial:h.update(scri_radial_closure='donor_seven_last3',scri_radial_bulk_rows=3)
        n = h['n']
        if h['admitted'] != 1 or h['kind'] not in KINDS:
            raise ValueError('Checkpoint is unadmitted or has an unknown physical kind')
        if n < 6 or n % 2 or h['cells'] != 19*n**3 or h['surface'] != 31*6*n*n:
            raise ValueError('Unsupported topology: expected fixed unrefined 19-block composite grid')
        if not math.isfinite(h['t']) or h['t'] < 0 or not math.isfinite(h['dt']) or h['dt'] <= 0:
            raise ValueError('Invalid checkpoint clock')
        if expected_time is not None:
            if not math.isfinite(expected_time) or abs(h['t']-expected_time) > 1e-12:
                raise ValueError('Checkpoint does not match requested comparison time')
        nv, ns = 31*h['cells'], h['surface']
        expected_size = h['header_bytes']+16*(nv+ns)
        if path.stat().st_size != expected_size:
            raise ValueError('Truncated checkpoint payload or trailing bytes')
        checksum = hashlib.sha256(raw)
        arrays = []
        for name, count in [('state', nv), ('compensation', nv),
                            ('boundary', ns), ('boundary-compensation', ns)]:
            offset = stream.tell()
            remaining = 8*count
            component_hash = hashlib.sha256()
            while remaining:
                block = stream.read(min(CHUNK, remaining))
                if not block or len(block) % 8:
                    raise ValueError('Checkpoint changed or became truncated during validation')
                if any(not math.isfinite(x[0]) for x in struct.iter_unpack('<d', block)):
                    raise ValueError(f'Nonfinite values in checkpoint {name}')
                checksum.update(block)
                component_hash.update(block)
                remaining -= len(block)
            arrays.append(dict(name=name, count=count, offset=offset,
                               sha256=component_hash.hexdigest()))
        sha = checksum.hexdigest()
        if expected_hash is not None and sha != expected_hash:
            raise ValueError('Checkpoint SHA256 does not match the expected input')
    h['magic'] = h['magic'].rstrip(b'\0').decode('ascii')
    return h, arrays, sha


def validate_report(path, h):
    report = read_json(path)
    if report.get('status') not in ['completed', 'checkpointed']:
        raise ValueError('Source report is not a completed or walltime-checkpointed run')
    for key in ['n', 'cells', 'steps', 't', 'dt']:
        if report.get(key) != h[key]:
            raise ValueError(f'Source report/checkpoint mismatch: {key}')
    if report.get('kind') != KINDS[h['kind']]:
        raise ValueError('Source report/checkpoint physical kind mismatch')
    scheme = 'none' if h['magic']=='HYP3D_MPI_V1' else 'final_stage_standard'
    angular = 'cross_face_centered5' if h['magic'] in ['HYP3D_MPI_V3','HYP3D_MPI_V4','HYP3D_MPI_V5','HYP3D_MPI_V6'] else 'face_local6'
    if (report.get('checkpoint_schema',h['magic']) != h['magic']
            or report.get('algebraic_projection','none') != scheme
            or report.get('scri_angular_stencil','face_local6') != angular):
        raise ValueError('Source report/checkpoint algebraic scheme mismatch')
    for key in ['scri_dissipation_epsilon','scri_dissipation_degree','checkpoint_header_bytes']:
        header_key='header_bytes' if key=='checkpoint_header_bytes' else key
        expected=h[header_key]
        value=report.get(key) if h['magic'] in ['HYP3D_MPI_V4','HYP3D_MPI_V5','HYP3D_MPI_V6'] else report.get(key,expected)
        if type(value) not in [int,float] or not math.isfinite(value) or value!=expected:
            raise ValueError('Source report/checkpoint dissipation configuration mismatch: '+key)
    if h['magic'] in ['HYP3D_MPI_V5','HYP3D_MPI_V6']:
        if report.get('checkpoint_schema')!=h['magic']:
            raise ValueError('V5 source report must state its checkpoint schema')
        for key in ['scri_normal_derivative','scri_normal_radial_nodes','scri_normal_bulk_derivative_order'] + (['scri_radial_closure','scri_radial_bulk_rows'] if h['magic']=='HYP3D_MPI_V6' else []):
            if type(report.get(key)) is not type(h[key]) or report[key]!=h[key]:
                raise ValueError('Source report/checkpoint normal configuration mismatch: '+key)
    for key in ['checkpoint_written', 'checkpoint_admitted']:
        if key in report and report[key] is not True:
            raise ValueError(f'Source report rejects {key}')
    return report


def verify_evidence(args, checkpoint_hash, report_hash):
    info = dict(execution_artifacts_verified=False, source_manifest_verified=False,
                source_files_verified=False)
    if args.worker_status:
        worker = read_json(args.worker_status)
        matches = []
        for case in worker.get('cases', []):
            if args.case and case.get('name') != args.case:
                continue
            artifacts = case.get('artifacts', {})
            check = [p for p, m in artifacts.items()
                     if p.endswith('.checkpoint') and m.get('sha256') == checkpoint_hash]
            reports = [p for p, m in artifacts.items()
                       if p.endswith('.report.json') and m.get('sha256') == report_hash]
            if len(check) == 1 and len(reports) == 1:
                matches.append((case, check[0], reports[0]))
        if len(matches) != 1:
            raise ValueError('Worker evidence must identify exactly one matching checkpoint/report case')
        case, checkpoint_name, report_name = matches[0]
        info.update(execution_artifacts_verified=True, case=case['name'],
                    case_status=case.get('status'), worker_status=worker.get('status'),
                    worker_status_sha256=digest(args.worker_status),
                    checkpoint_artifact=checkpoint_name, report_artifact=report_name,
                    source_manifest_sha256=worker.get('source_manifest_sha256'),
                    binaries=worker.get('binaries'), plan_sha256=worker.get('plan_sha256'))
    elif args.case:
        raise ValueError('--case requires --worker-status')
    if args.source_manifest:
        if not args.worker_status:
            raise ValueError('--source-manifest requires worker evidence to authenticate its hash')
        if digest(args.source_manifest) != info['source_manifest_sha256']:
            raise ValueError('Source manifest does not match the worker source binding')
        source = read_json(args.source_manifest)
        if not isinstance(source, dict) or not source:
            raise ValueError('Source manifest must contain relative file/hash pairs')
        info['source_manifest_verified'] = True
        for name, sha in source.items():
            if Path(name).is_absolute() or '..' in Path(name).parts or not isinstance(sha, str) \
                    or not re.fullmatch('[0-9a-f]{64}', sha):
                raise ValueError('Invalid source manifest entry')
            if args.source_root:
                path = (args.source_root/name).resolve()
                if not path.is_relative_to(args.source_root.resolve()) or digest(path) != sha:
                    raise ValueError(f'Source file differs from bound source: {name}')
        info['source_files_verified'] = args.source_root is not None
    elif args.source_root:
        raise ValueError('--source-root requires --source-manifest')
    return info


def export_array(stream, item, destination):
    """17 significant digits round-trip every finite binary64 value, including -0."""
    stream.seek(item['offset'])
    row = struct.Struct('<31d')
    count = item['count']//31
    with destination.open('x', newline='') as output:
        for _ in range(count):
            raw = stream.read(row.size)
            if len(raw) != row.size:
                raise ValueError('Checkpoint changed or became truncated during export')
            values = row.unpack(raw)
            output.write(','.join(format(x, '.17g') for x in values)+'\n')


def coordinates(n):
    """Same block/cell order as CompositeMesh; analytic coordinate-volume Jacobian."""
    length, radii = .125, (.45, .75, 1.)
    for block in range(19):
        layer, face = ((block-1)//6, (block-1) % 6) if block else (-1, -1)
        for k in range(n):
            for j in range(n):
                for i in range(n):
                    local = ((i+.5)/n, (j+.5)/n, (k+.5)/n)
                    if not block:
                        x = [-length+q*(2*length) for q in local]
                        volume = (2*length)**3/n**3
                    else:
                        s, u, v = local[0], -1+2*local[1], -1+2*local[2]
                        norm = math.sqrt(1+u*u+v*v)
                        if layer == 0:
                            d = (1-s)*length+s*radii[0]/norm
                            ds = radii[0]/norm-length
                        else:
                            dr = radii[layer]-radii[layer-1]
                            d = (radii[layer-1]+s*dr)/norm
                            ds = dr/norm
                        x = [0., 0., 0.]
                        axis = face//2
                        x[axis] = -d if face % 2 else d
                        x[(axis+1) % 3], x[(axis+2) % 3] = d*u, d*v
                        volume = 4*d*d*ds/n**3
                    yield (*x, math.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]), volume)


def compatible_report(h, source, provenance):
    # This status belongs to the completed export. Preserve source campaign
    # completion independently, including an accepted walltime checkpoint.
    target = source.get('target_time', h['t'])
    result = dict(status='completed', status_scope='snapshot export, not source campaign completion',
                snapshot_only=True, source_status=source['status'], source_target_time=target,
                source_target_reached=source['status'] == 'completed' and h['t'] >= target-1e-12,
                n=h['n'], cells=h['cells'], kind=KINDS[h['kind']], t=h['t'], dt=h['dt'],
                steps=h['steps'], t_over_M=3*h['t'] if h['kind'] == 2 else None,
                checkpoint_schema=h['magic'], checkpoint_header_bytes=h['header_bytes'],
                scri_dissipation_epsilon=h['scri_dissipation_epsilon'],
                scri_dissipation_degree=h['scri_dissipation_degree'],
                null_max=source.get('max_null_surface'), pole_max=source.get('max_pole'),
                compatibility_max=source.get('max_compatibility', source.get('compatibility_max')),
                native_fastflow={k: v for k, v in source.items()
                                 if k.startswith('horizon_') or k == 'diagnostic_time'},
                execution_provenance_verified=provenance['execution_artifacts_verified'])
    if h['magic'] in ['HYP3D_MPI_V5','HYP3D_MPI_V6']:
        for key in ['scri_normal_derivative','scri_normal_radial_nodes','scri_normal_bulk_derivative_order'] + (['scri_radial_closure','scri_radial_bulk_rows'] if h['magic']=='HYP3D_MPI_V6' else []):
            result[key]=h[key]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path, help='Fresh local directory, compatible with compare_coupled --runs')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--expected-time', type=float)
    parser.add_argument('--worker-status', type=Path)
    parser.add_argument('--case')
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--include-compensation', action='store_true')
    parser.add_argument('--reanalyze', type=Path, help='Existing local reanalyze executable; never runs evolution')
    parser.add_argument('--timeout', type=float, default=300.)
    parser.add_argument('--drop-intermediates', action='store_true')
    args = parser.parse_args()
    if args.drop_intermediates and not args.reanalyze:
        parser.error('--drop-intermediates requires successful --reanalyze')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be positive and finite')
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    args.report = (args.report or args.checkpoint.with_suffix('.report.json')).resolve()
    h, arrays, checkpoint_hash = validate_checkpoint(args.checkpoint, args.expected_sha256, args.expected_time)
    source = validate_report(args.report, h)
    report_hash = digest(args.report)
    provenance = verify_evidence(args, checkpoint_hash, report_hash)
    if args.reanalyze:
        args.reanalyze = args.reanalyze.resolve()
        analyzer_hash = digest(args.reanalyze)
    args.output.mkdir(parents=True, exist_ok=False)
    prefix = args.output/'evolution'
    manifest = dict(schema=1, status='exporting', checkpoint=str(args.checkpoint),
                    checkpoint_sha256=checkpoint_hash, source_report_sha256=report_hash,
                    header_bytes=h['header_bytes'], header=h, arrays=arrays, provenance=provenance,
                    momentum_conventions=MOMENTUM,
                    coordinate_measure='analytic Jacobian of the fixed unrefined composite map',
                    script_sha256=digest(Path(__file__)), output_files={}, removed_intermediates=[])
    atomic_json(args.output/'export.json', manifest)
    intermediates = []
    try:
        with args.checkpoint.open('rb') as stream:
            for item in arrays:
                if not args.include_compensation and 'compensation' in item['name']:
                    continue
                path = prefix.with_suffix('.'+item['name']+'.csv')
                export_array(stream, item, path)
                intermediates.append(path)
        # Prevent mixing two snapshots if the input changed while being exported.
        if digest(args.checkpoint) != checkpoint_hash:
            raise ValueError('Checkpoint changed during export')
        with prefix.with_suffix('.coordinates.csv').open('x', newline='') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(['x', 'y', 'z', 'r', 'volume'])
            writer.writerows(tuple(format(x, '.17g') for x in row) for row in coordinates(h['n']))
        shutil.copyfile(args.report, args.output/'source.report.json')
        for path, name in [(args.worker_status, 'source.worker-status.json'),
                           (args.source_manifest, 'source.manifest.json')]:
            if path:
                shutil.copyfile(path, args.output/name)
        if args.reanalyze:
            command = [str(args.reanalyze), str(h['n']), str(prefix), str(prefix)]
            radial=h['magic']=='HYP3D_MPI_V6'
            if radial:command.append('donor_seven')
            manifest['reanalyze'] = dict(command=command, executable_sha256=analyzer_hash,
                bulk_diagnostic_operator='donor_seven_last3' if radial else 'legacy_grid',
                ray_diagnostic_operator='interpolated_jets')
            with (args.output/'reanalyze.log').open('x') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            if result.returncode:
                raise RuntimeError(f'Diagnostic reanalysis exited {result.returncode}; see reanalyze.log')
            if radial and 'bulk_diagnostic_operator=donor_seven_last3 ray_diagnostic_operator=interpolated_jets' not in (args.output/'reanalyze.log').read_text().splitlines():
                raise RuntimeError('Reanalysis did not confirm the requested V6 diagnostic operator')
            for suffix in ['.diagnostics.csv', '.rays.csv']:
                if not prefix.with_suffix(suffix).is_file():
                    raise RuntimeError('Reanalysis did not create the expected diagnostics')
        output_report = compatible_report(h, source, provenance)
        output_report['diagnostics_generated'] = bool(args.reanalyze)
        if args.reanalyze:
            for key in ['bulk_diagnostic_operator','ray_diagnostic_operator']:
                output_report[key]=manifest['reanalyze'][key]
        atomic_json(prefix.with_suffix('.json'), output_report)
        for path in args.output.iterdir():
            if path.is_file() and path.name != 'export.json':
                manifest['output_files'][path.name] = dict(bytes=path.stat().st_size, sha256=digest(path))
        if args.drop_intermediates:
            for path in intermediates:
                path.unlink()
                manifest['removed_intermediates'].append(path.name)
        manifest['status'] = 'completed'
        atomic_json(args.output/'export.json', manifest)
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        atomic_json(args.output/'export.json', manifest)
        raise
    print(json.dumps(dict(status='completed', output=str(args.output), t=h['t'], n=h['n'],
                          checkpoint_sha256=checkpoint_hash, source_status=source['status'],
                          diagnostics_generated=bool(args.reanalyze), provenance=provenance), indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f'checkpoint export failed: {error}', file=sys.stderr)
        sys.exit(1)
