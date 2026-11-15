#!/usr/bin/env python3
"""Run native read-only checkpoint diagnostics on a bounded two-node allocation."""
import argparse
import json
import subprocess
from pathlib import Path
from hybrid_science_segment import header, sha


def main():
    p = argparse.ArgumentParser()
    for key in ['root', 'run']:
        p.add_argument('--'+key, type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--reference', type=Path)
    source.add_argument('--checkpoints-json', type=Path,
                        help='Explicit list of isolated science or completed archived legacy checkpoints; read only')
    p.add_argument('--wrapper', required=True)
    p.add_argument('--binding', required=True)
    a = p.parse_args()
    exe = a.root/'checkpoint-history-v4-portable/history_output'
    reducer = a.root/'checkpoint-harness-v4-source/tst/z4c_hyperboloidal/hybrid_checkpoint_audit.py'
    a.run.mkdir(exist_ok=False)
    results = []
    selected = []
    if a.reference:
        for i, mode in enumerate(['legacy', 'masked', 'agreement']):
            reference = a.reference/('%02d-%s-8' % (i, mode))
            assert json.loads((reference/'equivalence.json').read_text())['accepted_payload_bitwise_equal']
            checkpoint = max((reference/'rst').glob('*.rst'), key=lambda f: header(f)['cycle'])
            assert header(checkpoint)['cycle'] == 8
            selected.append((mode, mode, checkpoint))
    else:
        paths = json.loads(a.checkpoints_json.read_text())
        assert isinstance(paths, list) and 1 <= len(paths) <= 12
        for value in paths:
            checkpoint = Path(value).resolve()
            archived = False
            try:
                relative = checkpoint.relative_to((a.root/'science').resolve())
            except ValueError:
                # The user selected these completed campaigns as baselines.
                # Only diagnostics run here; no restart identity is rewritten.
                campaigns = {'headon-6M-20261109': 'N12', 'headon-6M-N16-20261110': 'N16'}
                relative = checkpoint.relative_to(a.root.parent.resolve())
                assert len(relative.parts) == 5 and relative.parts[0] in campaigns
                assert relative.parts[1] == 'runs' and relative.parts[3] == 'rst'
                n, mode = campaigns[relative.parts[0]], 'legacy'
                case, archived = n+'-legacy', True
            else:
                assert len(relative.parts) == 4 and relative.parts[2] == 'rst'
                case = relative.parts[0]
                n, mode = case.split('-', 1)
                assert n in ['N12', 'N16'] and mode in ['legacy', 'masked', 'agreement']
            metadata = header(checkpoint)
            # Existing terminal checkpoints overshot30 by less than one step.
            # New runs retain their strict <=30 cap.
            assert metadata['time_M0'] <= (30.002 if archived else 30)
            selected.append((case+'-cycle-%06d' % metadata['cycle'], mode, checkpoint))
        assert len({name for name, _, _ in selected}) == len(selected)
        # This version includes self-contained reducer/parser source snapshots.
        reducer = a.root/'science-analysis/hybrid_checkpoint_audit.py'
        (a.run/'checkpoints.json').write_bytes(a.checkpoints_json.read_bytes())
    for name, mode, checkpoint in selected:
        metadata = header(checkpoint)
        original_hash = sha(checkpoint)
        folder = a.run/name
        folder.mkdir()
        command = ['mpiexec', '-n', '24', '-ppn', '12', '--cpu-bind=list:'+a.binding,
                   a.wrapper, str(exe), str(checkpoint), 'checkpoint', '.25', '.18', '.195']
        provenance = {'command': command, 'checkpoint': str(checkpoint),
                      'checkpoint_header': metadata, 'checkpoint_sha256': original_hash,
                      'executable_sha256': sha(exe), 'reducer_sha256': sha(reducer),
                      'driver_sha256': sha(Path(__file__))}
        (folder/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        with (folder/'run.log').open('w') as f:
            result = subprocess.run(command, cwd=folder, stdout=f, stderr=subprocess.STDOUT, timeout=500)
        result.check_returncode()
        assert 'PASS checkpoint diagnostics; zero evolution steps' in (folder/'run.log').read_text()
        assert sha(checkpoint) == original_hash
        with (folder/'audit.log').open('w') as f:
            subprocess.run(['/usr/bin/python3', str(reducer), '--checkpoint', str(checkpoint),
                            '--history-dir', str(folder), '--output', str(folder/'audit')],
                           stdout=f, stderr=subprocess.STDOUT, check=True, timeout=300)
        results.append({'name': name, 'mode': mode, 'native_read_only_pass': True, 'original_checkpoint_unchanged': True,
                        'independent_Theta_volume_check_pass': True, 'provenance': provenance})
        (a.run/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        print('PASS native GPU and independent checkpoint audit: '+mode, flush=True)


if __name__ == '__main__':
    main()
