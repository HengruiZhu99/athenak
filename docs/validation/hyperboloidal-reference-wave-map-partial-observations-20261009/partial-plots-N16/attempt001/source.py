"""Plot already observed scalar JSON; no arrays, kernel or native advance."""
from pathlib import Path
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASES = (
    ('wave-map-N16-large-t2', 1.3671453700579306, 'Physical reference wave-map'),
    ('c0-N16-large-t2', 1.9994906249988413, 'Matched C0'),
)
PINS = {
    'wave-map-N16-large-t2': {
        'owner_rows': '07a78f495d72f5d55df29b1929bb8697115b3f61ca129c31adc1700d10f0b4d6',
        'owner_receipt': 'd591790541a6a5a928b19ef972a1a5f0420c00b02210a3ac32aa07f910073678',
        'root_rows': '1088352d843b915a7781a78680debfa51f99c3e326700ce6b3f2df0103416808',
        'native_receipt': '9b57613601546288c06b22a0c8ac34bc76fd1379c4224e3cf1706c5dc2d9b7e7',
        'native_stderr': '04d5cddc915421cdac62ef34b00ab0eb8a8597816414ab4d64580cb25a98974e',
    },
    'c0-N16-large-t2': {
        'owner_rows': '368b278387366a28fff0c7465166535077f5851903d2d79e6594ee07a6aa4b5b',
        'owner_receipt': '6cbd46728eac6f245d45caee8c1ad150f29c52f80b142e8522b607a89ca57040',
        'root_rows': 'aeaf02c4d943a28aa624d3e758681d02439cdd35783fdb7c4e4bce9280c7e696',
        'native_receipt': '4d69b3d78f2fc7db17bdf191f01486a76069cc63bf8dd2e23aa041442ddc2c5f',
        'native_stderr': '523d3cf650d8c5260f00237af4200048d61e9b6903a9666c6030bc3dc6da146f',
    },
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    value = json.loads(Path(path).read_text(), parse_constant=invalid)
    def finite(item):
        if isinstance(item, float):
            require(math.isfinite(item), 'Nonfinite scalar JSON')
        elif isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    return value


def main():
    attempt = HERE / 'attempt001'
    attempt.mkdir(exist_ok=False)
    started = time.monotonic()
    pins = {str(Path(__file__).resolve()): sha(__file__)}
    receipt = {'plot_protocol_completed': False, 'partial_diagnostic_only': True,
               'accepted_native_run': False, 'new_kernel_calls': 0, 'new_native_steps': 0,
               'scope': 'Saved scalar JSON and process metadata only; no differentiated '
                        'independent constraint calculation or abort-state reconstruction.',
               'command': sys.argv, 'cwd': str(Path.cwd()), 'python_version': sys.version,
               'environment': {key: os.environ.get(key) for key in
                               ['PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'MPLCONFIGDIR',
                                'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']}}
    try:
        (attempt / 'source.py').write_bytes(Path(__file__).read_bytes())
        data = []
        for case, abort_time, title in CASES:
            owner = ROOT / 'build-layer-research/reference-wave-map-partial-diagnostic-held-20261009/attempts' / (case + '-001')
            independent = ROOT / 'build-layer-research/wave-map-native-partial-fields-root-20261009/attempts' / (case + '-001')
            native = ROOT / 'build-layer-research/wave-map-native-t2-root-20261009/batch001' / case
            paths = {'owner_rows': owner / 'snapshot-observations.json',
                     'owner_receipt': owner / 'receipt.json',
                     'root_rows': independent / 'rows.json',
                     'native_receipt': native / 'launch-receipt.json',
                     'native_stderr': native / 'native.stderr'}
            for key, path in paths.items():
                require(sha(path) == PINS[case][key], 'Pin mismatch ' + str(path))
                pins[str(path)] = PINS[case][key]
            root_receipt_path = independent / 'receipt.json'
            pins[str(root_receipt_path)] = sha(root_receipt_path)
            rows, roots = load(paths['owner_rows']), load(paths['root_rows'])
            own_receipt, root_receipt, native_receipt = (load(paths['owner_receipt']),
                                                       load(root_receipt_path), load(paths['native_receipt']))
            require(own_receipt['accepted_native_run'] is False and root_receipt['accepted_native_run'] is False,
                    'Partial result mislabeled accepted')
            require(own_receipt['observer_completed'] and own_receipt['protected_before_after_equal']
                    and root_receipt['diagnostic_protocol_completed'] and root_receipt['before_after_equal'],
                    'Observer provenance failed')
            require(native_receipt['returncode'] == -6 and native_receipt['sources_before_after_equal'],
                    'Original failure/provenance differs')
            abort_match = re.search(r'mesh_time=([^ ]+) ', paths['native_stderr'].read_text())
            require(abort_match and float(abort_match.group(1)) == abort_time,
                    'Abort time differs from original stderr')
            require(root_receipt['rows_sha256'] == PINS[case]['root_rows'], 'Root row binding')
            require(own_receipt['observations_sha256'] == PINS[case]['owner_rows'], 'Owner row binding')
            require(len(rows) == len(roots) == own_receipt['saved_restart_files'], 'Row count differs')
            max_constraint_difference = 0.
            for row, root in zip(rows, roots):
                require(row['rst_sha256'] == root['sha256'] and row['time'] == root['time']
                        and row['cycle'] == root['cycle'], 'Snapshot identities differ')
                require(row['finite_extrema25'] == root['finite_extrema25'], 'Independent field extrema differ')
                require(root['saved_field_guards_satisfied'] and not row['diagnostic_guard_failures'],
                        'Saved diagnostic guard failed')
                for left, right in zip(row['native_diagnostics']['rms_H_Mcon_Zcon_Theta'],
                                       root['original_native_history_H_Mcon_Zcon_Theta']):
                    max_constraint_difference = max(max_constraint_difference, abs(left - right))
            require(max_constraint_difference <= 2e-11, 'Stored native history/probe discrepancy')
            require(rows[-1]['time'] < abort_time < 2., 'Abort/saved-time relation')
            require(all(a['time'] < b['time'] for a, b in zip(rows, rows[1:])), 'Unordered times')
            data.append((case, abort_time, title, rows, max_constraint_difference))
        dump(attempt / 'pins-before.json', pins)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        receipt['matplotlib_version'] = matplotlib.__version__
        fig, axes = plt.subplots(3, 2, figsize=(11.4, 10.2), sharex='col', constrained_layout=True)
        colors = ['#2864b4', '#c64d27', '#359057', '#8c4fad']
        labels = ['H', 'M', 'Z', 'Theta']
        summaries = []
        for col, (case, abort_time, title, rows, difference) in enumerate(data):
            times = [row['time'] for row in rows]
            axes[0, col].set_title(title + ' / N16', fontsize=12)
            for j, (label, color) in enumerate(zip(labels, colors)):
                positive = [(row['time'], row['native_diagnostics']['rms_H_Mcon_Zcon_Theta'][j])
                            for row in rows if row['native_diagnostics']['rms_H_Mcon_Zcon_Theta'][j] > 0]
                axes[0, col].semilogy([v[0] for v in positive], [v[1] for v in positive],
                                      color=color, label=label, linewidth=1.4)
                fractions = [sum(b['squared_fraction4'][j] for b in row['native_diagnostics']['radial_bins']
                                 if b['rlo'] >= .9) for row in rows]
                axes[2, col].plot(times, fractions, color=color, label=label, linewidth=1.4)
            for key, label, color, style in [
                ('alpha_min', 'lapse min', '#2864b4', '-'),
                ('chi_min', 'chi min', '#359057', '-'),
                ('minimum_conformal_metric_eigenvalue', 'g-tilde eigenvalue min', '#8c4fad', '-'),
            ]:
                axes[1, col].plot(times, [row[key] for row in rows], label=label, color=color, linestyle=style)
            axes[1, col].plot(times, [row['finite_extrema25'][18]['max'] for row in rows],
                              label='lapse max', color='#2864b4', linestyle='--')
            for ax in axes[:, col]:
                ax.axvline(abort_time, color='#b02a37', linestyle=':', linewidth=1.3)
                ax.set_xlim(0., 2.)
                ax.grid(True, alpha=.2)
            axes[0, col].text(.03, .96, 'Native abort t=' + format(abort_time, '.6f'),
                              transform=axes[0, col].transAxes, va='top', color='#b02a37')
            axes[0, col].set_ylabel('Constraint RMS')
            axes[1, col].set_ylabel('Saved field extrema')
            axes[2, col].set_ylabel('Squared error fraction at r >= 0.9')
            axes[2, col].set_ylim(0., 1.02)
            axes[2, col].set_xlabel('Coordinate time (S=1, a=0.5, M=0)')
            for ax in axes[:, col]:
                ax.legend(fontsize=8, loc='best')
            summaries.append({'case': case, 'saved_arrays': len(rows), 'last_saved_time': times[-1],
                              'native_abort_time_from_pinned_stderr': abort_time,
                              'probe_vs_stored_native_history_max_absolute_difference': difference,
                              'independent_extrema_exactly_equal': True})
        fig.suptitle('Failed native runs: saved snapshots only\nFinal saved states precede the unrecorded abort stages', fontsize=14)
        fig.savefig(attempt / 'partial-N16-diagnostics.png', dpi=160)
        fig.savefig(attempt / 'partial-N16-diagnostics.pdf')
        plt.close(fig)
        require(all(sha(path) == digest for path, digest in pins.items()), 'Input drift during plot')
        dump(attempt / 'pins-after.json', pins)
        receipt.update(plot_protocol_completed=True, protected_before_after_equal=True, cases=summaries,
                       outputs=[{'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
                                for path in sorted(attempt.glob('partial-N16-diagnostics.*'))])
    except BaseException as error:
        (attempt / 'failure.stderr').write_text(traceback.format_exc())
        receipt.update(error=repr(error), protected_before_after_equal=False)
    receipt['seconds'] = time.monotonic() - started
    dump(attempt / 'receipt.json', receipt)
    print(json.dumps(receipt, allow_nan=False))
    return 0 if receipt['plot_protocol_completed'] else 1


if __name__ == '__main__':
    sys.exit(main())
