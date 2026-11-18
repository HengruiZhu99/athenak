"""Place measured tile traffic on two six-GPU Aurora nodes."""
from functools import lru_cache
from itertools import combinations
import re


def _sibling_bytes(tiles, matrix):
    return sum(matrix[a][b] + matrix[b][a] for a, b in zip(tiles[::2], tiles[1::2]))


def _pair_tiles(source, target, matrix):
    """Assign twelve source ranks to fixed sibling slots on their selected node."""
    count = len(source)

    @lru_cache(maxsize=None)
    def assign(remaining):
        if not remaining:
            return 0, 0, ()
        available = [i for i in range(count) if remaining & (1 << i)]
        slot = count - len(available)
        best = None
        for i, j in combinations(available, 2):
            a, b = source[i], source[j]
            neg_bytes, rest_moves, rest = assign(remaining ^ (1 << i) ^ (1 << j))
            # Either orientation gives the same traffic. Keep as many labels as possible.
            moves, pair = min(((x != target[slot]) + (y != target[slot + 1]), (x, y))
                              for x, y in ((a, b), (b, a)))
            candidate = (neg_bytes - matrix[a][b] - matrix[b][a],
                         rest_moves + moves, pair + rest)
            if best is None or candidate < best:
                best = candidate
        return best

    neg_bytes, _, order = assign((1 << count) - 1)
    # Keep the preceding node placement when a different matching only ties it.
    return order if -neg_bytes > _sibling_bytes(source, matrix) else source


def reorder_nodes(owners, receipt):
    ranks = 24
    binding = receipt['rank_binding']
    if len(binding)!=ranks or receipt['nodes']!=2:
        raise ValueError('two-node rank binding is required for measured node placement')
    devices = {}
    for rank in range(ranks):
        host,tile = binding[str(rank)]
        gpu,lane = map(int,tile.split('.'))
        devices.setdefault((host,gpu),{})[lane]=rank
    if len(devices)!=12 or any(set(lanes)!={0,1} for lanes in devices.values()):
        raise ValueError('node placement needs twelve physical GPUs with both tiles')
    groups=sorted(devices)
    if groups[0][0]!=groups[5][0] or groups[6][0]!=groups[-1][0] or groups[0][0]==groups[-1][0]:
        raise ValueError('expected six GPUs per node')
    matrix={}
    for line in receipt['work']:
        row=re.fullmatch(r'MULTIRATE_PEER_BYTES cycle=1 rank=(\d+) bytes=([\d,]+)',line)
        if row:
            matrix[int(row[1])]=list(map(int,row[2].split(',')))
    if set(matrix)!=set(range(ranks)) or any(len(row)!=ranks for row in matrix.values()):
        raise ValueError('incomplete first-interval measured traffic matrix')
    wire=int(next(re.search(r' sent_bytes=(\d+)',s)[1] for s in receipt['work']
                  if s.startswith('MULTIRATE_WORK cycle=1 ')))
    if sum(sum(row) for row in matrix.values())!=wire:
        raise ValueError('per-peer traffic disagrees with the interval wire counter')
    traffic = {(a,b):sum(matrix[r][s]+matrix[s][r] for r in devices[a].values()
                         for s in devices[b].values()) for a,b in combinations(groups,2)}
    def cut(first):
        return sum(volume for (a,b),volume in traffic.items() if (a in first)!=(b in first))
    current=frozenset(groups[:6])
    # Node exchange is symmetric. Anchor the first physical GPU to remove ties.
    best=min((frozenset((groups[0],*subset)) for subset in combinations(groups[1:],5)),
             key=lambda x:(cut(x),len(x.symmetric_difference(current)),sorted(x)))
    order=sorted(best)+[g for g in groups if g not in best]
    target = [devices[g][lane] for g in groups for lane in (0, 1)]
    source = [devices[g][lane] for g in order for lane in (0, 1)]
    # Older receipts without the initial finest-level census can still reorder whole GPUs.
    census = next((line for line in receipt['work']
                   if line.startswith('MULTIRATE_PLACEMENT cycle=0 ')), None)
    equal_counts = False
    if census is not None:
        counts = re.search(r'\bblocks_by_rank=([\d,]+)$', census)
        if not counts or len(counts[1].split(',')) != ranks:
            raise ValueError('incomplete finest-level block counts')
        equal_counts = len(set(map(int, counts[1].split(',')))) == 1
    if equal_counts:
        for offset in (0, ranks // 2):
            node = slice(offset, offset + ranks // 2)
            source[node] = _pair_tiles(source[node], target[node], matrix)
    placement = dict(zip(source, target))

    return ([placement[rank] for rank in owners],
            dict(wire_bytes=wire,original_internode_bytes=cut(current),candidate_internode_bytes=cut(best),
                 original_intragpu_bytes=_sibling_bytes(target, matrix),
                 candidate_intragpu_bytes=_sibling_bytes(source, matrix),
                 equal_finest_tile_counts=equal_counts,
                 rank_permutation=[placement[r] for r in range(ranks)]))


def group_finest_on_sibling_tiles(owners, levels, kernels):
    """A fixed-map N12 experiment. Keep each physical GPU's original blocks."""
    from collections import Counter

    if len(owners)!=len(levels) or len(levels)!=len(kernels) or set(owners)!=set(range(24)):
        raise ValueError('expected the full two-node N12 placement')
    finest=max(levels)
    groups=[(level,kernel) for level,kernel in zip(levels,kernels)]
    result=list(owners)
    def batches(histogram):
        return sum(4*(1<<level) for (level,_),count in histogram.items() if count)
    def work(histogram):
        return sum(count*(1<<level) for (level,_),count in histogram.items())
    for first in range(0,24,2):
        slots=[[i for i,r in enumerate(owners) if r==rank] for rank in (first,first+1)]
        fine=[[i for i in ids if levels[i]==finest] for ids in slots]
        if any(len(ids)!=1 for ids in fine):
            raise ValueError('expected exactly one finest block per tile in the spread control')
        if groups[fine[0][0]]!=groups[fine[1][0]]:
            continue  # Scri and ordinary kernels cannot form one batch.
        counts=[Counter(groups[i] for i in ids) for ids in slots]
        before=sum(map(batches,counts))
        best=None
        for source in (0,1):
            fast=fine[source][0]
            for slow in slots[1-source]:
                if levels[slow]==finest:
                    continue
                left,right=(h.copy() for h in counts)
                for h,sign in ((left,1-2*source),(right,2*source-1)):
                    h[groups[fast]]-=sign
                    h[groups[slow]]+=sign
                after=batches(left)+batches(right)
                if after>=before:
                    continue
                candidate=(after,abs(work(left)-work(right)),fast,slow)
                if best is None or candidate<best:
                    best=candidate
        if best:
            fast,slow=best[-2:]
            result[fast],result[slow]=result[slow],result[fast]
    if Counter(result)!=Counter(owners) or any(a//2!=b//2 for a,b in zip(owners,result)):
        raise AssertionError('sibling exchange changed physical placement or tile capacity')
    return result


def sibling_trial(control):
    import json
    from pathlib import Path
    import struct
    import sys

    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'z4c_hyperboloidal'))
    from restart_roundtrip import parse

    row=json.loads(Path(control).read_text())
    run=Path(row['log']).parent
    header=parse(Path(row['checkpoint']))
    atlas=dict(field.split('=',1) for field in header['atlas'].decode().split(';')[1:])
    def dimensions(key):
        return list(map(int,atlas[key].strip(',').split(',')))
    radial=sum(dimensions('radial_blocks'))
    root_level=(max(radial,*dimensions('core'),*dimensions('angular'))-1).bit_length()
    kernels=[]
    for x,_,_,level,chart in struct.iter_unpack('=5i',header['topology']):
        kernels.append('cartesian' if chart==0 else
                       'scri' if x+1==radial*(1<<(level-root_level)) else 'spherical')
    matches=re.findall(r'graph fingerprint=\d+ levels=([0-9,]+)',(run/'run.log').read_text())
    if len(matches)!=1 or row['regrids']:
        raise ValueError('expected one fixed hierarchy in the sibling control')
    levels=list(map(int,matches[0].split(',')))
    with (run/'partition.log').open('rb') as stream:
        lines=stream.read(int(stream.readline())).decode().splitlines()
        if stream.read().strip():
            raise ValueError('control changed its placement')
    ids,owners=zip(*(map(int,line.split()) for line in lines[1:]))
    if list(ids)!=list(range(header['blocks'])):
        raise ValueError('noncanonical partition record')
    new=group_finest_on_sibling_tiles(owners,levels,kernels)
    def batches(mapping):
        return sum(4*(1<<level) for _,level,_ in set(zip(mapping,levels,kernels)))
    observed={int(re.search(r' batches=(\d+)',line)[1]) for line in row['work']
              if line.startswith('MULTIRATE_WORK cycle=')}
    if observed!={2*batches(owners)}:
        raise ValueError('reconstructed kernel groups disagree with the executed batches')
    return dict(original=list(owners),owners=new,original_batches=2*batches(owners),
                predicted_batches=2*batches(new),moved_blocks=sum(a!=b for a,b in zip(owners,new)))


if __name__=='__main__':
    import json
    import sys
    print(json.dumps(sibling_trial(sys.argv[1])))
