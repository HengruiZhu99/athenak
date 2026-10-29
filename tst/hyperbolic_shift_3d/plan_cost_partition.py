#!/usr/bin/env python3
"""Propose fixed contiguous cuts from measured bulk RHS cost; never run a job.

Rank-average cost is a piecewise-constant estimate, not a per-cell profile.
The predicted imbalance must be checked by a new same-node experiment.
"""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path


def plan(report, ranks):
    if report.get('status') != 'completed' or report.get('timing_per_rank_exclusive') is not True:
        raise ValueError('Require a completed run with exclusive rank timings')
    if report.get('checkpoint_schema') != 'HYP3D_MPI_V11' or report.get('segment_steps', 0) < 1:
        raise ValueError('Require an evolved V11 trajectory')
    cells, old_ranks, n = (report.get(k) for k in ('cells', 'mpi_size', 'n'))
    if any(type(x) is not int or x < 1 for x in (cells, old_ranks, n, ranks)) or max(ranks, old_ranks)>cells:
        raise ValueError('Invalid grid or rank count')
    # Future measured partitions must report their actual cuts. Historical V11
    # reports before that field existed used the fixed equal-cell split.
    old_cuts = report.get('ownership_cuts', [cells*r//old_ranks for r in range(old_ranks+1)])
    if (len(old_cuts)!=old_ranks+1 or old_cuts[0]!=0 or old_cuts[-1]!=cells
            or any(type(x) is not int for x in old_cuts)
            or any(a>=b for a,b in zip(old_cuts,old_cuts[1:]))):
        raise ValueError('Invalid measured ownership intervals')
    phases=report.get('phase_seconds_by_rank')
    if not isinstance(phases,list) or len(phases)!=old_ranks:
        raise ValueError('Missing rank timing coverage')
    if any(not isinstance(row,list) or len(row)!=5 or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in row) for row in phases):
        raise ValueError('Invalid exclusive timings')
    costs=[row[0] for row in phases]
    if min(costs)<=0:
        raise ValueError('Bulk RHS cost must be positive on every rank')
    cumulative=[0.]
    for cost in costs:cumulative.append(cumulative[-1]+cost)
    if not math.isfinite(cumulative[-1]):raise ValueError('Total cost overflow')
    def prefix(point):
        if point==cells:return cumulative[-1]
        r=bisect.bisect_right(old_cuts,point)-1
        return cumulative[r]+costs[r]*(point-old_cuts[r])/(old_cuts[r+1]-old_cuts[r])
    cuts=[0]
    for rank in range(1,ranks):
        target=cumulative[-1]*rank/ranks
        r=min(old_ranks-1,bisect.bisect_right(cumulative,target)-1)
        position=old_cuts[r]+(target-cumulative[r])*(old_cuts[r+1]-old_cuts[r])/costs[r]
        lower=cuts[-1]+1;upper=cells-(ranks-rank)
        candidates={min(upper,max(lower,math.floor(position))),min(upper,max(lower,math.ceil(position)))}
        cuts.append(min(candidates,key=lambda x:(abs(prefix(x)-target),x)))
    cuts.append(cells)
    estimated=[prefix(b)-prefix(a) for a,b in zip(cuts,cuts[1:])]
    return dict(n=n,cells=cells,ranks=ranks,cuts=cuts,source_ranks=old_ranks,
                source_cuts=old_cuts,source_bulk_rhs_seconds=costs,
                estimated_bulk_rhs_seconds=estimated,
                source_max_over_mean=max(costs)/(sum(costs)/old_ranks),
                estimated_max_over_mean=max(estimated)/(sum(estimated)/ranks),
                scope='Fixed contiguous cuts using rank-average bulk RHS cost only; ignores communication and boundary work. Prediction, not measured speedup.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--ranks',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=args.report.read_bytes();result=plan(json.loads(data),args.ranks)
    result['source_report_sha256']=hashlib.sha256(data).hexdigest()
    sidecar=args.output.with_name(args.output.name+'.json')
    if args.output.exists() or sidecar.exists():raise ValueError('Use fresh output paths')
    text='HYP3D_COST_PARTITION_V1\n{} {} {}\n{}\n{}\n'.format(result['n'],result['cells'],result['ranks'],result['source_report_sha256'],' '.join(map(str,result['cuts'])))
    args.output.write_text(text);sidecar.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
