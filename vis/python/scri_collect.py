#!/usr/bin/env python3
"""Collect complete native scri records into a resumable surface-metric cache.

This cache is temporary analysis data, not a replacement for the original
archives. All source records and epoch files remain read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import scri_reader as reader
from scri_geometry import endpoint_jets, spacetime_metric


def collect(root, output, capacity=4096, record_pattern="segment-*/*.o20.*.r00000000.scri"):
    output.mkdir(parents=True, exist_ok=True)
    index_file = output/'records.json'
    records = json.loads(index_file.read_text()) if index_file.exists() else []
    known = {r['cycle']: r for r in records}
    epochs = {}; grid = None
    identity_file = output/'identity.json'
    identity = json.loads(identity_file.read_text()) if identity_file.exists() else None
    def epoch_for(source, state):
        nonlocal identity
        if state.get('kind') != 'native scri accepted state':
            raise ValueError('expected native accepted-state record')
        epoch_path = source.parent/state['epoch']
        if epoch_path not in epochs:
            epoch = reader.read(epoch_path)
            if epoch.get('kind') != 'native scri topology epoch':
                raise ValueError('expected native topology epoch')
            stamp = {key: epoch[key] for key in ('initial_data_identity', 'fields', 'units', 'basis')}
            stamp['source_manifest_sha256'] = hashlib.sha256(epoch['source_manifest'].encode()).hexdigest()
            for key in ('configuration', 'boundary_source', 'conformal_profile'):
                stamp[key] = epoch[key].tolist()
            if identity is None:
                identity = stamp
                identity_file.write_text(json.dumps(identity, indent=2)+'\n')
            elif stamp != identity:
                raise ValueError('mixed evolution/source identity in analysis cache')
            epochs[epoch_path] = epoch
        return epochs[epoch_path]
    if records:
        grid = dict(np.load(output/'grid.npz'))
        metric = np.load(output/'metric.npy', mmap_mode='r+')
    def incomplete(error):
        return isinstance(error, FileNotFoundError) or str(error) in ('truncated scri archive', 'invalid array extent')
    sources = []
    candidates = sorted(root.glob(record_pattern),
                        key=lambda p: (int(p.name.rsplit('.', 3)[1]), p.parent.name))
    for source_index, path in enumerate(candidates):
        try:
            state = reader.read(path)
        except (ValueError, OSError) as error:
            if source_index == len(candidates)-1 and incomplete(error):
                break
            raise
        sources.append((int(state['record'][0]), path, state))
    sources.sort(key=lambda row: row[0])
    for ordinal, (cycle, path, first) in enumerate(sources):
        epoch_for(path, first)
        if cycle in known:
            if float(first['time'][0]) != known[cycle]['time']:
                raise ValueError('duplicate cycle has inconsistent time')
            continue
        if records and cycle <= records[-1]['cycle']:
            raise ValueError('cannot insert a late-discovered record into an existing cache')
        count = int(first['record'][4]); parts = []
        try:
            for rank in range(count):
                source = path.with_name(path.name.replace('r00000000.scri', f'r{rank:08}.scri'))
                state = first if rank == 0 else reader.read(source)
                if not np.array_equal(state['record'][[0, 1, 2, 4, 5, 6, 7]], first['record'][[0, 1, 2, 4, 5, 6, 7]]) or int(state['record'][3]) != rank:
                    raise ValueError('mixed rank/record identifiers')
                if not np.array_equal(state['time'], first['time']):
                    raise ValueError('mixed rank times')
                parts.append((state, epoch_for(source, state)))
        except (ValueError, OSError) as error:
            if ordinal != len(sources)-1 or not incomplete(error):
                raise
            break  # Retry the final incomplete MPI output on the next collection.
        ids = np.concatenate([e['surface_ids'][:, [0, 2, 3]] for _, e in parts])
        order = np.lexsort(ids.T[::-1]); ids = ids[order]
        if len(np.unique(ids, axis=0)) != len(ids):
            raise ValueError('duplicate native surface nodes')
        x = np.concatenate([e['surface_geometry'][:, :3] for _, e in parts])[order]
        weights = np.concatenate([e['surface_weights'] for _, e in parts])[order]
        if grid is None:
            grid = dict(ids=ids, x=x, weights=weights)
            np.savez(output/'grid.npz', **grid)
            metric = np.lib.format.open_memmap(output/'metric.npy', mode='w+', dtype='f8',
                                               shape=(capacity, len(x), 4, 4, 13))
        else:
            for key, value in [('ids', ids), ('x', x), ('weights', weights)]:
                if not np.array_equal(grid[key], value):
                    raise ValueError('analysis cache requires a fixed native surface topology')
        if len(records) == len(metric):
            raise ValueError('analysis cache capacity exhausted')
        jets = np.concatenate([endpoint_jets(s, e) for s, e in parts])[order]
        restored = spacetime_metric(jets, x)
        if not np.isfinite(restored).all():
            raise ValueError('nonfinite reconstructed surface metric')
        metric[len(records)] = restored
        record = dict(cycle=cycle, time=float(first['time'][0]), source=str(path), ranks=count)
        records.append(record); known[cycle] = record
        metric.flush()
        temporary = index_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(records, indent=2)+'\n'); temporary.replace(index_file)
        if len(records) % 100 == 0:
            print('Collected', len(records), 'records; native t=', record['time'], flush=True)
    print('Complete cached records:', len(records), 'last native t:', records[-1]['time'] if records else None, flush=True)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capacity', type=int, default=4096)
    parser.add_argument('--record-pattern',default='segment-*/*.o20.*.r00000000.scri')
    args = parser.parse_args()
    collect(args.root.resolve(), args.output.resolve(), args.capacity, args.record_pattern)
