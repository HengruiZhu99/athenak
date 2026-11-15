#!/usr/bin/env python3
"""Read-only comparison of isolated hybrid head-on histories and checkpoint audits.

Accepts partial runs explicitly. Norm ratios at two resolutions are indicators,
not a fitted convergence order or a proof of nonlinear stability.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

MODES = ['legacy', 'masked', 'agreement']
COLORS = dict(zip(MODES, ['#555555', '#176f9c', '#bc582d']))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge(paths, time_column=0):
    rows = {}
    receipts = {}
    for path in sorted(paths):
        receipts[str(path)] = sha(path)
        raw = [s for s in path.read_text().splitlines() if s.strip() and not s.startswith('#')]
        if not raw:
            continue
        data = np.atleast_2d(np.loadtxt(raw))
        if not np.isfinite(data).all():
            raise ValueError('nonfinite history: ' + str(path))
        for row in data:
            # Restart-boundary rows are retained once, from the later segment.
            rows[float(row[time_column])] = row
    return np.array([rows[t] for t in sorted(rows)]), receipts


def growth(history, column):
    time = history[:, 0] * 3
    use = (time >= 26) & (time <= 30) & (history[:, column] > 0)
    if np.count_nonzero(use) < 4:
        return None
    x, y = time[use], np.log(history[use, column])
    slope, intercept = np.polyfit(x, y, 1)
    variance = np.sum((y - y.mean())**2)
    return {'samples': len(x), 'interval_M0': [float(x[0]), float(x[-1])],
            'log_slope_per_M0': float(slope),
            'efolding_M0': float(1/slope) if slope > 0 else None,
            'R2': float(1-np.sum((y-slope*x-intercept)**2)/variance) if variance > 0 else None}


def checkpoint_plots(report, output):
    """Fixed physical regions are independent of horizon-exterior masks."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout='constrained')
    corner_fig, corner_axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    any_audit = False
    for name, case in report['cases'].items():
        audits = case['checkpoint_audits']
        if not audits:
            continue
        any_audit = True
        n, mode = name.split('-', 1)
        style = dict(color=COLORS[mode], ls='-' if n == 'N12' else '--',
                     marker='.', label=name)
        times = np.array([a['time_M0'] for a in audits])
        edges = audits[0]['fixed_band_edges']
        assert all(a['fixed_band_edges'] == edges for a in audits)
        sums = np.array([a['fixed_band_sums'] for a in audits])
        assert np.all(sums[:, :, 0] > 0) and np.all(sums[:, :, 1] >= 0)
        rms = np.sqrt(sums[:, :, 1]/sums[:, :, 0])/3
        case['fixed_band_Theta'] = {'times_M0': times.tolist(), 'edges': edges,
                                  'M0_Theta_rms': rms.tolist()}
        for b, ax in enumerate(axes.flat):
            ax.semilogy(times, np.ma.masked_less_equal(rms[:, b], 0), **style)
            ax.set(title='Compact radius %.2f–%.2f' % (edges[b], edges[b+1]),
                   xlabel=r'$t/M_0$', ylabel=r'$M_0$ Theta RMS')
            ax.grid(alpha=.2)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        corner_rows = []
        for a in audits:
            # This symptom region is exactly one shell in the matched geometry.
            radii = np.array(a['shell_radii'])
            layers = np.flatnonzero(np.isclose(radii[:-1], .32) & np.isclose(radii[1:], .40))
            assert len(layers) == 1
            layer = np.array(a['shell_sums'])[layers[0]]
            samples = np.array([s['native_Theta'] for s in a['common_coordinate_samples']])
            corner_rows.append([np.sqrt(layer[3]/layer[2])/3,
                                layer[3]/layer[1] if layer[1] > 0 else 0,
                                np.sqrt(np.mean(samples**2))/3])
        corner_rows = np.array(corner_rows)
        case['corner_Theta'] = {'times_M0': times.tolist(),
                               'columns': ['M0_corner_rms', 'corner_fraction_of_shell_Theta2', 'M0_common_sample_rms'],
                               'rows': corner_rows.tolist(),
                               'region': 'r=0.32..0.40, both |u|,|v|>=0.75; common samples at r=.34,.36,.38 and u=v=±.85 on six faces'}
        for ax, col in [(corner_axes[0, 0], 0), (corner_axes[1, 0], 2)]:
            ax.semilogy(times, np.ma.masked_less_equal(corner_rows[:, col], 0), **style)
        corner_axes[0, 1].plot(times, corner_rows[:, 1], **style)
        counts = np.array([a['all_evolved_and_scri_cumulative_chi_counts'] for a in audits])
        # Linear axis includes genuine zero counts without an invented floor.
        corner_axes[1, 1].plot(times, counts[:, 0], **style)
    if not any_audit:
        plt.close(fig); plt.close(corner_fig)
        return
    for ax, title, ylabel in zip(corner_axes.flat,
            ['Shell-corner Theta, r=0.32–0.40', 'Corner share of shell Theta squared',
             'Identical-coordinate Theta samples', 'Cumulative chi-floor events'],
            [r'$M_0$ RMS', 'Fraction', r'$M_0$ sample RMS', 'All evolved and scri slots']):
        ax.set(title=title, xlabel=r'$t/M_0$', ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    corner_axes[0, 0].legend(fontsize=7)
    suffix = '' if report['required_cases_complete'] else ' — partial'
    fig.suptitle('Fixed-radius checkpoint constraints'+suffix)
    corner_fig.suptitle('Corner and clipping diagnostics'+suffix)
    fig.savefig(output/'fixed-bands.png', dpi=180)
    corner_fig.savefig(output/'corners-and-chi.png', dpi=180)
    plt.close(fig); plt.close(corner_fig)


def native_fixed_band_plots(report, output):
    for label, col, scale in [('H', 1, 1/9), ('M', 2, 1/9), ('Z', 4, 1/3)]:
        fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout='constrained')
        found = False
        for name, case in report['cases'].items():
            audits = [a for a in case['checkpoint_audits'] if a['native_history_check']]
            if not audits:
                continue
            found = True
            times = [a['time_M0'] for a in audits]
            integrals = np.array([[r['native_raw_integrals'] for r in a['native_history_check']]
                                  for a in audits])
            assert integrals.shape == (len(audits), 8, 5)
            assert np.isfinite(integrals).all() and np.all(integrals >= 0)
            assert np.all(integrals[:, :, 0] > 0)
            rms = np.sqrt(integrals[:, :, col]/integrals[:, :, 0])*scale
            edges = audits[0]['fixed_band_edges']
            assert all(a['fixed_band_edges'] == edges for a in audits)
            case.setdefault('native_fixed_band_constraints', {})[label] = {
                'times_M0': times, 'edges': edges, 'normalized_rms': rms.tolist()}
            n, mode = name.split('-', 1)
            for b, ax in enumerate(axes.flat):
                ax.semilogy(times, np.ma.masked_less_equal(rms[:, b], 0),
                            color=COLORS[mode], ls='-' if n == 'N12' else '--', marker='.', label=name)
                ax.set(title='Compact radius %.2f–%.2f' % (edges[b], edges[b+1]),
                       xlabel=r'$t/M_0$', ylabel=(r'$M_0^2$ RMS' if col < 3 else r'$M_0$ RMS'))
                ax.grid(alpha=.2)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        if found:
            axes[0, 0].legend(fontsize=7)
            fig.suptitle('Fixed-radius '+label+('' if report['required_cases_complete'] else ' — partial'))
            fig.savefig(output/('fixed-bands-'+label+'.png'), dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('science', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy-archive', type=Path,
                        help='Verified archived legacy receipt tree; prioritizes agreement at both resolutions')
    parser.add_argument('--modes', nargs='+', choices=MODES, default=MODES,
                        help='Schemes to display; optional mask-only controls can be omitted')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'Fresh matched head-on histories; partial until all six cases reach their fixed cycle limits.',
              'script_sha256': sha(Path(__file__)), 'cases': {}, 'resolution_checks': {},
              'agreement_vs_legacy': {},
              'limitations': ['Two-resolution constraint-norm ratios are not a convergence-order measurement.',
                              'Horizon-exterior masks can change with horizon behavior; fixed-radius bands must be assessed separately.',
                              'Native segment finalization can add off-cadence rows and skip the next scheduled history sample on restart; comparisons use recorded times.',
                              'Scri D-Lambda and RHS maxima are amplitudes, not zero-target constraint errors.']}
    required_modes = args.modes
    if args.legacy_archive:
        required_modes = ['legacy', 'agreement']
        report['scope'] = ('New N12/N16 mask-plus-agreement versus completed archived legacy campaigns; '
                           'partial mask-only controls are supplementary. No legacy campaign is resumed.')
        report['limitations'].append('Archived terminal steps slightly exceed30M0; source times are preserved and late fits use only t<=30M0.')
    curves = {}
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    overlap_figure, overlap_axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    scri_figure, scri_axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    tracker_figure, tracker_axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    for n in [12, 16]:
        for mode in args.modes:
            name = 'N%d-%s' % (n, mode)
            case = args.science/name
            archived = None
            if mode == 'legacy' and args.legacy_archive:
                case = args.legacy_archive/name
                archive_path = case/'archived-case.json'
                archived = json.loads(archive_path.read_text())
                assert archived['kind'] == 'archived_legacy' and archived['n'] == n
                assert archived['complete'] and 30 <= archived['completed_time_M0'] < 30.002
            segments = sorted(case.glob('segment-*'))
            histories, receipts = merge(case.glob('segment-*/*.z4c_hyperboloidal.hst'))
            if not len(histories):
                continue
            assert histories.shape[1] == 12
            assert histories[-1, 0]*3 <= (30.002 if archived else 30)
            curves[name] = histories
            provenance = [json.loads((d/'provenance.json').read_text()) for d in segments
                          if (d/'provenance.json').exists()]
            if archived:
                provenance = [archived]
                receipts[str(archive_path)] = sha(archive_path)
            else:
                for key in ['input_sha256', 'executable_sha256', 'initial_data_identity']:
                    assert len({r[key] for r in provenance}) == 1, (name, key)
            completed = [json.loads((d/'segment-result.json').read_text()) for d in segments
                         if (d/'segment-result.json').exists()]
            row = {'end_M0': float(histories[-1, 0]*3), 'complete': bool(archived) or any(r['complete'] for r in completed),
                   'source_kind': 'archived_legacy' if archived else 'fresh_isolated',
                   'provenance': provenance, 'history_files': receipts,
                   'native_dt_values': np.unique(histories[:, 1]).tolist(),
                   'last_constraint_row_native': histories[-1].tolist(),
                   'exterior_growth_26_30': {label: growth(histories, col) for label, col in
                                           zip(['H', 'M', 'Theta', 'Z'], [8, 9, 10, 11])}}
            row['evolution_failures'] = []
            for segment in segments:
                exit_file = segment/'exit-code.txt'
                if exit_file.exists() and int(exit_file.read_text().strip()) != 0:
                    log = segment/'run.log'
                    messages = [s.strip() for s in log.read_text().splitlines() if 'what():' in s]
                    row['evolution_failures'].append({
                        'segment': str(segment), 'native_launcher_exit_code': int(exit_file.read_text().strip()),
                        'exit_file_sha256': sha(exit_file), 'run_log_sha256': sha(log),
                        'exception_messages': messages})
            row['failed'] = not row['complete'] and any(
                failure['segment'] == str(segments[-1]) for failure in row['evolution_failures'])
            horizon_status = []
            for log in case.glob('segment-*/run.log'):
                for line in log.read_text().splitlines():
                    if not line.startswith('binary horizon component='):
                        continue
                    values = dict(re.findall(r'(\w+)=([^ ]+)', line))
                    horizon_status.append(values)
            row['horizon_status_counts'] = {
                'samples': len(horizon_status),
                'incomplete': sum(r.get('complete') != '1' for r in horizon_status),
                'above_diagnostic_expansion_threshold': sum(r.get('expansion_above_diagnostic_threshold') == '1' for r in horizon_status),
                'acceptance_policies': sorted({r.get('acceptance', 'unspecified') for r in horizon_status})}
            scri, files = merge(case.glob('segment-*/*.scri.hst'))
            assert len(scri) and scri.shape[1] == 22
            row['scri'] = {'files': files, 'maximum_native_columns_3_through_22': np.max(abs(scri[:, 2:]), axis=0).tolist()}
            style = dict(color=COLORS[mode], ls='-' if n == 12 else '--', label=name)
            for ax, col, label in zip(scri_axes.flat, [2, 3, 8, 9, 19, 21],
                    ['Compatibility', 'Clock', 'Theta', 'det(g)−1', 'D−Lambda amplitude', 'RHS amplitude']):
                ax.semilogy(scri[:, 0]*3, np.ma.masked_less_equal(abs(scri[:, col]), 0), **style)
                ax.set(title=label, xlabel=r'$t/M_0$', ylabel='Native maximum')
                ax.grid(alpha=.2)
            for component in [0, 1]:
                track, files = merge(case.glob('segment-*/*.co_%d.txt' % component), 1)
                row['tracker_%d' % component] = {'files': files, 'rows': len(track),
                                                'last': track[-1].tolist() if len(track) else None}
                horizon, files = merge(case.glob('segment-*/*.horizon_summary_%d.txt' % component), 1)
                row['horizon_%d' % component] = {'files': files, 'rows': len(horizon),
                                                'last': horizon[-1].tolist() if len(horizon) else None}
                if len(track):
                    assert track.shape[1] == 8
                    tracker_axes[component, 0].plot(track[:, 1]*3, track[:, 2]*3, **style)
                if len(horizon):
                    assert horizon.shape[1] == 12
                    tracker_axes[component, 1].plot(horizon[:, 1]*3, horizon[:, 2]*3, **style)
                    # The historical 'hrms' column stores mean square (FastFlow
                    # ah_prop[hhrms]); the native surface summary takes sqrt.
                    assert np.all(horizon[:, 8] >= 0)
                    tracker_axes[component, 2].semilogy(horizon[:, 1]*3,
                        np.ma.masked_less_equal(np.sqrt(horizon[:, 8]), 0), **style)
            if mode != 'legacy':
                disagreement, files = merge(case.glob('segment-*/*.overlap.hst'))
                assert len(disagreement) and disagreement.shape[1] == 14
                assert np.all(disagreement[:, 8] <= 0)
                if mode == 'masked':
                    assert np.all(disagreement[:, 8] == 0)
                row['overlap'] = {'files': files, 'last_native': disagreement[-1].tolist()}
                for col, ax, label in zip(range(2, 8), overlap_axes.flat,
                                          ['Chi', 'Metric', 'Curvature', 'Lambda', 'Gauge', 'D-Lambda']):
                    ax.semilogy(disagreement[:, 0]*3, np.ma.masked_less_equal(disagreement[:, col], 0),
                                color=COLORS[mode], ls='-' if n == 12 else '--', label=name)
                    ax.set(title=label, xlabel=r'$t/M_0$', ylabel='Native disagreement RMS')
                    ax.grid(alpha=.2)
            audit_paths = sorted(case.glob('audits/*/audit.json'))
            audits = [json.loads(p.read_text()) for p in audit_paths]
            assert all(a['n'] == n and a['time_M0'] <= (30.002 if archived else 30) for a in audits)
            row['checkpoint_audit_files'] = {str(p): sha(p) for p in audit_paths}
            # A later native diagnostic pass supersedes the independent-only
            # reduction at the same checkpoint cycle; retain both input hashes.
            unique = {}
            for a in sorted(audits, key=lambda x: bool(x['native_history_check'])):
                unique[a['cycle']] = a
            row['checkpoint_audits'] = sorted(unique.values(), key=lambda x: x['time'])
            if mode == 'legacy':
                observed = [a['native_checkpoint_overlap']['native_row']
                            for a in row['checkpoint_audits'] if a.get('native_checkpoint_overlap')]
                if observed:
                    observed = np.array(observed)
                    for col, ax in zip(range(2, 8), overlap_axes.flat):
                        ax.semilogy(observed[:, 0]*3, np.ma.masked_less_equal(observed[:, col], 0),
                                    color=COLORS[mode], ls='-' if n == 12 else '--', marker='.',
                                    label=name+' checkpoints')
            report['cases'][name] = row
            for ax, col, label, scale in zip(axes.flat, [8, 9, 10, 11], ['H', 'M', 'Theta', 'Z'], [1/9, 1/9, 1/3, 1/3]):
                ax.semilogy(histories[:, 0]*3, np.ma.masked_less_equal(histories[:, col]*scale, 0),
                            color=COLORS[mode], ls='-' if n == 12 else '--',
                            label=name+(' (failed)' if row['failed'] else ''))
                if row['failed']:
                    ax.plot(histories[-1, 0]*3, histories[-1, col]*scale,
                            marker='x', ms=8, mew=2, color=COLORS[mode])
                ax.set(title='Horizon-exterior '+label, xlabel=r'$t/M_0$',
                       ylabel=(r'$M_0^2$ RMS' if col < 10 else r'$M_0$ RMS'))
                ax.grid(alpha=.2)
    for n in [12, 16]:
        names = ['N%d-%s' % (n, mode) for mode in ['legacy', 'agreement']]
        if not all(name in curves for name in names):
            continue
        legacy, agreement = [curves[name] for name in names]
        rows = []
        for point in agreement:
            time = point[0]
            if not (0 < time*3 <= 30 and legacy[0, 0] <= time <= legacy[-1, 0]):
                continue
            reference = np.array([np.interp(time, legacy[:, 0], legacy[:, col])
                                  for col in range(8, 12)])
            if np.all(reference > 0):
                rows.append([float(time*3), *(point[8:12]/reference).tolist()])
        report['agreement_vs_legacy']['N%d' % n] = {
            'columns': ['time_M0', 'H_ratio', 'M_ratio', 'Theta_ratio', 'Z_ratio'],
            'rows': rows, 'last': rows[-1] if rows else None,
            'legacy_history_linearly_interpolated_to_agreement_times': True,
            'interpretation': 'Below1 means lower horizon-exterior RMS; assess fixed regions and horizon behavior separately.'}
    for mode in args.modes:
        if not all('N%d-%s' % (n, mode) in curves for n in [12, 16]):
            continue
        low, high = [curves['N%d-%s' % (n, mode)] for n in [12, 16]]
        common = high[(high[:, 0] >= low[0, 0]) & (high[:, 0] <= low[-1, 0])]
        selected = common[(common[:, 0] >= 6) & (common[:, 0]*3 <= 30)]
        rows = []
        for point in selected:
            coarse = np.array([np.interp(point[0], low[:, 0], low[:, col]) for col in range(8, 12)])
            if np.all(point[8:12] > 0):
                rows.append([float(point[0]*3), *(coarse/point[8:12]).tolist()])
        report['resolution_checks'][mode] = {'columns': ['time_M0', 'H_N12_over_N16', 'M_N12_over_N16', 'Theta_N12_over_N16', 'Z_N12_over_N16'],
                                            'rows': rows, 'coarse_history_linearly_interpolated_to_fine_times': True}
    assert report['cases'], 'no science histories found'
    report['initial_data_identities_by_resolution'] = {}
    for n in [12, 16]:
        identities = {v['provenance'][0]['initial_data_identity'] for k, v in report['cases'].items()
                      if k.startswith('N%d-' % n)}
        assert len(identities) <= 1, 'initial data differ between schemes at fixed resolution'
        report['initial_data_identities_by_resolution'][str(n)] = sorted(identities)
    report['all_six_complete'] = len(report['cases']) == 6 and all(r['complete'] for r in report['cases'].values())
    report['required_cases'] = ['N%d-%s' % (n, mode) for n in [12, 16] for mode in required_modes]
    report['required_cases_complete'] = all(report['cases'].get(name, {}).get('complete', False)
                                            for name in report['required_cases'])
    report['acceptance'] = 'Requires scientific assessment and qualified paired performance; no automatic acceptance from finite histories.'
    checkpoint_plots(report, args.output)
    native_fixed_band_plots(report, args.output)
    axes[0, 0].legend(fontsize=8, ncol=2)
    overlap_axes[0, 0].legend(fontsize=8)
    figure.suptitle('Matched head-on constraints' + ('' if report['required_cases_complete'] else ' — partial'))
    overlap_figure.suptitle('Overlap disagreement at fixed kappa M0 = 1/3')
    figure.savefig(args.output/'constraints.png', dpi=180)
    overlap_figure.savefig(args.output/'disagreement.png', dpi=180)
    scri_axes[0, 0].legend(fontsize=7)
    suffix = '' if report['required_cases_complete'] else ' — partial'
    scri_figure.suptitle('Scri diagnostics (native units)'+suffix)
    for component in [0, 1]:
        for col, label, ylabel in zip(range(3), ['Puncture x', 'Diagnostic surface mass', 'Expansion RMS'],
                                     [r'$x/M_0$', r'$M/M_0$', 'Native RMS']):
            tracker_axes[component, col].set(title='Component %d: %s' % (component, label),
                                            xlabel=r'$t/M_0$', ylabel=ylabel)
            tracker_axes[component, col].grid(alpha=.2)
        tracker_axes[component, 1].ticklabel_format(axis='y', style='plain', useOffset=False)
    tracker_axes[0, 0].legend(fontsize=7)
    tracker_figure.suptitle('Trajectories and diagnostic surfaces'+suffix)
    scri_figure.savefig(args.output/'scri.png', dpi=180)
    tracker_figure.savefig(args.output/'trajectories-and-surfaces.png', dpi=180)
    plots = ['constraints.png', 'disagreement.png', 'scri.png', 'trajectories-and-surfaces.png']
    if report['agreement_vs_legacy']:
        ratio_figure, ratio_axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
        for name, comparison in report['agreement_vs_legacy'].items():
            rows = np.array(comparison['rows'])
            if not len(rows):
                continue
            for col, ax in enumerate(ratio_axes.flat, 1):
                ax.plot(rows[:, 0], rows[:, col], color='#bc582d' if name == 'N12' else '#176f9c',
                        ls='-' if name == 'N12' else '--', label=name)
                ax.plot(rows[-1, 0], rows[-1, col], 'o', ms=4,
                        color='#bc582d' if name == 'N12' else '#176f9c')
        for ax, label in zip(ratio_axes.flat, ['H', 'M', 'Theta', 'Z']):
            ax.axhline(1, color='#777777', lw=1, ls=':')
            ax.set(title='Horizon-exterior '+label, xlabel=r'$t/M_0$', ylabel='New / legacy RMS')
            ax.grid(alpha=.2)
        ratio_axes[0, 0].legend()
        ratio_figure.suptitle('Mask + agreement / legacy at matched times — below 1 means lower RMS')
        ratio_figure.savefig(args.output/'constraint-ratios.png', dpi=180)
        plt.close(ratio_figure)
        plots.append('constraint-ratios.png')
    if any(case['checkpoint_audits'] for case in report['cases'].values()):
        plots += ['fixed-bands.png', 'corners-and-chi.png']
    if any(case.get('native_fixed_band_constraints') for case in report['cases'].values()):
        plots += ['fixed-bands-H.png', 'fixed-bands-M.png', 'fixed-bands-Z.png']
    report['plot_sha256'] = {name: sha(args.output/name) for name in plots}
    (args.output/'assessment.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: {'end_M0': v['end_M0'], 'complete': v['complete']} for k, v in report['cases'].items()}, indent=2))


if __name__ == '__main__':
    main()
