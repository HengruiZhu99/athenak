#!/usr/bin/env python3
"""Verify restart-aware history analysis using saved native 4+2 step outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from scri_checkpoint_receipt import receipt
from scri_merge_segments import combine, merge
from scri_run_summary import history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-verification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.native_verification.resolve(), args.output.resolve()
    assert json.loads((source/'results.json').read_text())['B_restart_repartition']['bitwise_equal']
    out.mkdir(parents=True, exist_ok=False)
    starts = [source/'B-on/rst/single_hole_20M.00000.rst',
              source/'B-on/rst/single_hole_20M.00001.rst']
    finals = [starts[1], source/'B-restart/rst/single_hole_20M.00002.rst']
    for i, name in enumerate(['B-on', 'B-restart']):
        d = out/name; d.mkdir()
        for f in (source/name).glob('*'):
            if f.is_file(): shutil.copy2(f, d/f.name)
        # Analysis fixtures only: the native numerical test already succeeded.
        (d/'exit-code.txt').write_text('0\n')
        (d/'executable.sha256').write_text('analysis-fixture-same-executable athena\n')
        for label, checkpoint in [('start', starts[i]), ('final', finals[i])]:
            (d/(label+'-checkpoint.json')).write_text(json.dumps(receipt(checkpoint), indent=2)+'\n')
    result = merge([out/'B-on', out/'B-restart'], out/'merged')
    assert json.loads((out/'merged/start-checkpoint.json').read_text()) == receipt(starts[0])
    assert json.loads((out/'merged/final-checkpoint.json').read_text()) == receipt(finals[-1])
    diffs = {}
    for name, record in result['streams'].items():
        merged, _ = history(out/'merged'/name)
        full, _ = history(source/'B-full6'/name)
        assert np.array_equal(merged[:, 0], full[:, 0])
        delta = float(np.max(abs(merged-full)))
        # Native MPI reductions can differ in their last bits across rank counts.
        np.testing.assert_allclose(merged, full, rtol=2e-12, atol=2e-13)
        assert all(row['complete_window'] for row in record['rows'])
        diffs[name] = delta

    # A restored state can predate the last copied history. Retain only the
    # authoritative new branch and flag its shortened first extrema window.
    a = np.array([[0., 1., 10.], [1., 1., 11.], [2., 1., 12.], [3., 1., 13.]])
    b = np.array([[2., 1., 22.], [3., 1., 23.], [4., 1., 24.]])
    rows, origins, removed = combine([a, b], [{'time_native':0.}, {'time_native':1.5}], True)
    np.testing.assert_array_equal(rows[:, 0], [0., 1., 2., 3., 4.])
    np.testing.assert_array_equal(rows[:, 2], [10., 11., 22., 23., 24.])
    assert removed == [0, 2] and not origins[2]['complete_window']
    assert origins[2]['window_start_native'] == 1.5 and origins[3]['complete_window']
    # A restart startup snapshot must not replace an accumulated extrema window.
    rows, origins, _ = combine([a[:2], a[1:]], [{'time_native':0.}, {'time_native':1.}], True)
    np.testing.assert_array_equal(rows, a)
    assert all(row['complete_window'] for row in origins)
    # Provenance guard: identical time/identity alone is insufficient.
    path = out/'B-restart/start-checkpoint.json'
    correct = path.read_text(); changed = json.loads(correct); changed['sha256'] = 'wrong'
    path.write_text(json.dumps(changed))
    try:
        merge([out/'B-on', out/'B-restart'], out/'must-not-exist')
    except AssertionError as error:
        assert 'restart chain mismatch' in str(error)
    else:
        raise AssertionError('mismatched checkpoint chain accepted')
    finally:
        path.write_text(correct)
    assert not (out/'must-not-exist').exists()
    report = dict(native_4plus2_vs_continuous6_max_absolute_difference=diffs,
                  native_aligned_windows_complete=True, rollback_supersedes_old_rows=True,
                  partial_restart_window_flagged=True, startup_extrema_ignored=True,
                  wrong_checkpoint_hash_rejected=True,
                  original_initial_and_last_final_receipts_preserved=True,
                  fixture_metadata='Exit/executable receipts wrap already-verified native outputs; no evolution launched.',
                  sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [Path(__file__), Path(__file__).with_name('scri_merge_segments.py'),
                            Path(__file__).with_name('scri_checkpoint_receipt.py')]})
    (out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
