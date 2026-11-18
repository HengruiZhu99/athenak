"""Map twelve measured physical-GPU assignments onto two six-GPU Aurora nodes."""
from itertools import combinations
import re


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
    placement={devices[old][lane]:devices[new][lane] for old,new in zip(order,groups) for lane in (0,1)}
    return ([placement[rank] for rank in owners],
            dict(wire_bytes=wire,original_internode_bytes=cut(current),candidate_internode_bytes=cut(best),
                 rank_permutation=[placement[r] for r in range(ranks)]))
