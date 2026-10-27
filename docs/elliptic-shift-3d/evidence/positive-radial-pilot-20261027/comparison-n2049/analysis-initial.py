#!/usr/bin/env python3
"""Compare frozen positive-storage pilots without omitting cells or relabeling failures.

This is bounded analysis only: no solver, evolution or reference run is launched.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

FIELDS = ('alpha', 'chi', 'g_rr', 'A_rr', 'Khat', 'Theta', 'Lambda_r', 'beta_r')
LABELS = {1: 'Direct positive', 2: 'Log/reference', 3: 'Square-root'}
REGIONS = (
    ('all', 'all', lambda r: r >= 0),
    ('puncture_r_lt_01', 'r<.1', lambda r: r < .1),
    ('bulk_r_01_03', '.1-.3', lambda r: (r >= .1) & (r < .3)),
    ('transition_r_03_08', '.3-.8', lambda r: (r >= .3) & (r < .8)),
    ('focused_shell_r_062_070', '.62-.70', lambda r: (r >= .62) & (r < .70)),
    ('outer_r_ge_08', 'r>=.8', lambda r: r >= .8),
    ('scri_r_ge_09', 'r>=.9', lambda r: r >= .9),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_path(path):
    if path.exists():
        return path
    compressed = path.with_name(path.name + '.gz')
    if compressed.exists():
        return compressed
    raise FileNotFoundError(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--reference-n', type=int, default=2049)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    output = args.output or args.pilot_root / f'comparison-n{args.reference_n}'
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f'Refusing to overwrite comparison evidence: {output}')
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    # Keep cache writes inside a temporary writable directory, not the user's
    # scientific checkout or their global matplotlib/font configuration.
    with tempfile.TemporaryDirectory(prefix='positive-radial-analysis-') as cache:
        os.environ.setdefault('MPLCONFIGDIR', cache)
        os.environ.setdefault('XDG_CACHE_HOME', cache)
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from radial_reference import interpolate

        def norms(error, weights):
            scale = float(np.max(np.abs(error)))
            rms = scale * float(np.sqrt(np.average((error / scale)**2, weights=weights))) if scale else 0.
            return dict(weighted_rms=rms, point_max=scale,
                        signed_weighted_bias=float(np.average(error, weights=weights)))

        def checked_csv(path):
            data = np.genfromtxt(path, delimiter=',', names=True)
            if not data.shape or not all(np.all(np.isfinite(data[name])) for name in data.dtype.names):
                raise ValueError(f'Empty or nonfinite input; no rows filtered: {path}')
            return data

        command_path = args.pilot_root / 'commands.json'
        commands = json.loads(command_path.read_text())
        reference_path = args.reference_root / f'radial-n{args.reference_n}.npz'
        reference_manifest_path = args.reference_root / 'manifest.json'
        reference_manifest = json.loads(reference_manifest_path.read_text())
        reference = np.load(reference_path, allow_pickle=False)
        if len(reference['r']) != args.reference_n or not np.all(np.diff(reference['r']) > 0):
            raise ValueError('Independent reference grid mismatch')
        input_hashes = {str(p): digest(p) for p in (command_path, reference_path, reference_manifest_path)}
        extension_path = args.reference_root / 'reference-extension.json'
        if extension_path.exists():
            input_hashes[str(extension_path)] = digest(extension_path)
        summary = dict(
            method='all-cell physical-volume comparison plus raw angular extraction; no evolution averaging',
            scientific_scope='single coarse N8 t=.0002 pilot; all recorded coupled cases fail; no convergence or collapse safety claim',
            reference=dict(n=args.reference_n, time=reference_manifest['time'],
                           method=reference_manifest['method'], mass=reference_manifest['mass']),
            region_note='all contains every cell; the remaining diagnostic regions overlap and never replace all-cell acceptance',
            input_hashes=input_hashes, modes={})
        coordinate_reference = None
        for mode in (1, 2, 3):
            case = next(c for c in commands['cases'] if c['mode'] == mode)
            stdout_path = args.pilot_root / f'mode{mode}.stdout'
            result = json.loads(stdout_path.read_text())
            if result['positive_storage'] != mode or abs(result['time'] - reference_manifest['time']) > 1e-15:
                raise ValueError('Mode/time mismatch with independent reference')
            data_path = physical_path(args.pilot_root / f'mode{mode}.csv')
            sphere_path = physical_path(args.pilot_root / f'mode{mode}.csv.spheres.csv')
            for path in (stdout_path, data_path, sphere_path):
                input_hashes[str(path)] = digest(path)
            data = checked_csv(data_path)
            spheres = checked_csv(sphere_path)
            if len(data) != result['cells'] or not np.all(data['weight'] > 0):
                raise ValueError('Cell/positive physical-volume weight mismatch')
            coordinates = np.column_stack([data[name] for name in ('x', 'y', 'z', 'r')])
            if coordinate_reference is None:
                coordinate_reference = coordinates
            elif not np.array_equal(coordinate_reference, coordinates):
                raise ValueError('The storage modes do not use the same matched physical cells')
            truth = {f: interpolate(reference['r'], reference[f], data['r']) for f in FIELDS}
            errors = {f: data[f] - truth[f] for f in FIELDS}
            row = dict(label=LABELS[mode], recorded_exit_code=case['exit_code'],
                       recorded_scientific_status='failed' if case['exit_code'] else 'passed_as_recorded',
                       pilot_result=result, all_cells=len(data), volume_quadrature=float(sum(data['weight'])),
                       matched_time_field_errors={f: norms(errors[f], data['weight']) for f in FIELDS},
                       regions={}, raw_angular_comparison={})
            for f in ('alpha', 'chi'):
                if not np.all(truth[f] > 0):
                    raise ValueError('Positive-reference relative diagnostic requires positive physical reference')
                row['matched_time_field_errors'][f]['relative_to_positive_reference'] = norms(errors[f] / truth[f], data['weight'])
            for name, _, select in REGIONS:
                mask = select(data['r'])
                if not np.any(mask):
                    row['regions'][name] = dict(cells=0, unavailable=True)
                    continue
                weights = data['weight'][mask]
                row['regions'][name] = dict(cells=int(sum(mask)), volume_quadrature=float(sum(weights)),
                    H=norms(data['H'][mask], weights), M=norms(data['M_norm'][mask], weights),
                    field_errors={f: norms(errors[f][mask], weights) for f in FIELDS})
            # Check that the raw CSV reproduces the pilot's full-domain norms;
            # this validates weighting/field interpretation, not scientific acceptance.
            for channel in ('H', 'M'):
                recomputed = row['regions']['all'][channel]['weighted_rms']
                reported = result[channel + '_rms']
                if abs(recomputed - reported) > 5e-12 * max(1., abs(reported)):
                    raise ValueError(f'Full-domain {channel} norm did not reproduce stdout')
            radii = np.unique(spheres['r'])
            for f in FIELDS:
                mean, scatter, minimum, maximum, angular_rms = [], [], [], [], []
                radial_truth = interpolate(reference['r'], reference[f], radii)
                counts = []
                for r, exact in zip(radii, radial_truth):
                    selection = spheres['r'] == r
                    values = spheres[f][selection]
                    directions = spheres['direction'][selection]
                    if len(values) != len(np.unique(directions)) or len(values) != 64:
                        raise ValueError('Raw angular sample layout changed; no directions removed')
                    counts.append(len(values));mean.append(float(np.mean(values)));scatter.append(float(np.std(values)))
                    minimum.append(float(np.min(values)));maximum.append(float(np.max(values)))
                    angular_rms.append(float(np.sqrt(np.mean((values - exact)**2))))
                row['raw_angular_comparison'][f] = dict(radii=radii.tolist(), directions_per_radius=counts,
                    reference=radial_truth.tolist(), mean=mean, scatter=scatter, minimum=minimum, maximum=maximum,
                    mean_error=(np.array(mean) - radial_truth).tolist(), angular_rms_error=angular_rms,
                    maximum_scatter=max(scatter))
            row['original_vector_residual'] = dict(
                source='pilot-reported original-equation stage maxima; not recomputed from derivative-incomplete CSV',
                physical_volume_rms_max=result['shift_residual_max'], point_max=result.get('shift_residual_point_max'))
            row['boundary_gates'] = {}
            for name, target in [('scri_tensor_pole', 1e-7), ('scri_ghost_clock', 1e-13),
                                 ('scri_ghost_null', 1e-12), ('scri_ghost_tangent', 1e-12)]:
                row['boundary_gates'][name] = dict(value=result[name], unchanged_target=target, passes=result[name] <= target)
            summary['modes'][str(mode)] = row

        coarse_path = args.reference_root / 'radial-n1025.npz'
        if coarse_path.exists() and args.reference_n != 1025:
            coarse = np.load(coarse_path, allow_pickle=False)
            input_hashes[str(coarse_path)] = digest(coarse_path)
            summary['independent_reference_control'] = dict(
                comparison=f'N1025 minus N{args.reference_n} at every matched 3D cell; mode1 supplied weights',
                errors={f: norms(interpolate(coarse['r'], coarse[f], data['r']) - truth[f],
                                 checked_csv(physical_path(args.pilot_root / 'mode1.csv'))['weight']) for f in FIELDS})
        colors = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green'}
        fig, axes = plt.subplots(4, 2, figsize=(11, 12), layout='constrained')
        for ax, f in zip(axes.flat, FIELDS):
            ax.plot(reference['r'], reference[f], color='black', lw=1.2, label=f'Independent spherical N{args.reference_n}')
            for mode in (1, 2, 3):
                angular = summary['modes'][str(mode)]['raw_angular_comparison'][f]
                r = np.array(angular['radii']); mean = np.array(angular['mean']); scatter = np.array(angular['scatter'])
                ax.plot(r, mean, 'o-', ms=2.5, color=colors[mode], label=LABELS[mode])
                ax.fill_between(r, mean - scatter, mean + scatter, color=colors[mode], alpha=.12)
            ax.set(xlabel='compact radius', ylabel=f);ax.grid(alpha=.22)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle('t=0.0002, N8: angular means and raw scatter; all three coupled pilots fail')
        fig.savefig(output / 'positive-storage-radial-comparison.png', dpi=160);plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
        x = np.arange(len(FIELDS))
        for mode in (1, 2, 3):
            row = summary['modes'][str(mode)]
            axes[0, 0].semilogy(x, [row['matched_time_field_errors'][f]['weighted_rms'] for f in FIELDS], 'o-', color=colors[mode], label=LABELS[mode])
            axes[0, 1].semilogy(x, [row['raw_angular_comparison'][f]['maximum_scatter'] for f in FIELDS], 'o-', color=colors[mode])
            for ax, channel in zip(axes[1], ('H', 'M')):
                ax.semilogy(np.arange(len(REGIONS)), [row['regions'][name][channel]['weighted_rms'] for name, _, _ in REGIONS], 'o-', color=colors[mode])
        for ax in axes[0]:
            ax.set_xticks(x, FIELDS, rotation=35, ha='right');ax.grid(alpha=.22)
        axes[0, 0].set_title('All-cell physical-volume RMS error against 1D');axes[0, 0].legend(fontsize=8)
        axes[0, 1].set_title('Maximum raw angular scatter across sampled radii')
        for ax, channel in zip(axes[1], ('H', 'M')):
            ax.set_xticks(np.arange(len(REGIONS)), [label for _, label, _ in REGIONS], rotation=35, ha='right')
            ax.set_title(f'{channel} RMS: all cells and overlapping diagnostic regions');ax.grid(alpha=.22)
        fig.suptitle('Positive-storage controls: no spatial/time convergence or collapse-safety claim')
        fig.savefig(output / 'positive-storage-errors-and-regions.png', dpi=160);plt.close(fig)
        summary['provenance'] = dict(command=sys.argv, analysis_script_sha256=digest(__file__),
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
            seconds=time.monotonic() - start, plots=['positive-storage-radial-comparison.png', 'positive-storage-errors-and-regions.png'])
        (output / 'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        concise = {mode: dict(status=row['recorded_scientific_status'],
            field_rms={f: row['matched_time_field_errors'][f]['weighted_rms'] for f in ('alpha', 'chi', 'A_rr', 'beta_r')},
            H_rms=row['regions']['all']['H']['weighted_rms'], M_rms=row['regions']['all']['M']['weighted_rms'],
            scri_tensor_pole=row['boundary_gates']['scri_tensor_pole']) for mode, row in summary['modes'].items()}
        print(json.dumps(concise, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
