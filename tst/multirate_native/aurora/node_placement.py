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
