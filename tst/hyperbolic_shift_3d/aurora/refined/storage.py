#!/usr/bin/env python3
"""Task storage accounting, including prior performance outputs; never deletes."""
import json
import os
from pathlib import Path
import sys

# User raised the allowance to 50 GB and authorized CompactBinaryMerger/hzhu
# scratch on 2026-11-03. This supersedes this task's earlier 10 GB guard only.
# BN1 is independent and is neither counted here nor modified.
LIMIT_BYTES = 50_000_000_000
PRIOR = Path('/home/hzhu/hyperboloidal-performance-scaling-20261103')
SCRATCH_PARENT = Path('/lus/flare/projects/CompactBinaryMerger/hzhu')


def footprint(roots):
    total, entries, seen = 0, 0, set()
    for root in roots:
        for directory, children, names in os.walk(root, followlinks=False):
            paths = [Path(directory), *[Path(directory) / n for n in names],
                     *[Path(directory) / n for n in children
                       if (Path(directory) / n).is_symlink()]]
            for path in paths:
                info = path.lstat()
                key = (info.st_dev, info.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                total += max(info.st_size, getattr(info, 'st_blocks', 0) * 512)
                entries += 1
    return dict(bytes=total, entries=entries)


def require_capacity(root, planned=0, reserve=2_000_000_000):
    root = Path(root).resolve()
    if root.parent != SCRATCH_PARENT or not root.name.startswith('hyperboloidal-refined-'):
        raise ValueError('Use a dedicated hyperboloidal-refined-* scratch directory')
    if planned < 0 or reserve < 0:
        raise ValueError('Negative storage reservation')
    measured = footprint([PRIOR, root])
    result = dict(measured, roots=[str(PRIOR), str(root)], limit_bytes=LIMIT_BYTES,
                  planned_bytes=planned, reserve_bytes=reserve,
                  headroom_bytes=LIMIT_BYTES-measured['bytes']-planned-reserve)
    if result['headroom_bytes'] < 0:
        raise RuntimeError('Task storage allowance exceeded: '+json.dumps(result))
    return result


if __name__ == '__main__':
    print(json.dumps(require_capacity(sys.argv[1], int(sys.argv[2]) if len(sys.argv)>2 else 0), indent=2))
