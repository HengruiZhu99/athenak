#!/usr/bin/env python3
"""Recompute per-rank scri timing comparisons; never launch a simulation."""
import csv
import hashlib
import json
import math
from pathlib import Path

root=Path(__file__).resolve().parent
verification=json.loads((root/'verification.json').read_text())
artifacts={entry['path']:entry for entry in verification['artifacts']}
reports=[]
for name in ['benchmark-staged','benchmark-inline']:
    path='output/'+name+'.report.json'
    assert hashlib.sha256((root/path).read_bytes()).hexdigest()==artifacts[path]['sha256']
    report=json.loads((root/path).read_text())
    assert report['status']=='completed' and report['benchmark_passed'] and report['placement_passed']
    assert report['backend']=='SYCL' and report['mpi_size']==12
    assert len(report['ranks'])==12
    assert [r['rank'] for r in report['ranks']]==list(range(12))
    reports.append(report)
staged,inline=reports
assert staged['sampling']=='staged' and inline['sampling']=='inline'
assert staged['benchmark_config']==inline['benchmark_config']
assert staged['source_manifest_sha256']==inline['source_manifest_sha256']
assert staged['plan_sha256']==inline['plan_sha256']
plan=json.loads((root/'plan.json').read_text())
names=[c['name'] for c in plan['commands']]
assert names.index('benchmark-staged')<names.index('benchmark-inline')
rows=[]
for a,b in zip(staged['ranks'],inline['ranks']):
    rank=a['rank']
    assert a['host']==b['host'] and a['mask']==b['mask']==f'{rank//2}.{rank%2}'
    assert a['hierarchy']==b['hierarchy']=='COMPOSITE'
    x,y=a['result']['metrics'],b['result']['metrics']
    for r,m in [(a,x),(b,y)]:
        assert r['result']['passed'] and r['result']['backend']=='SYCL'
        assert (m['n'],m['ranks'],m['rank'],m['benchmark_repeats'])==(16,12,rank,20)
        assert (m['first_row'],m['owned_rows'],m['sampled_fields'])==(128*rank,128,25)
        assert m['repeat_error']==0
        for key in ['evaluate_seconds_mean','evaluate_seconds_total','rhs_norm2','surface_norm2']:
            assert math.isfinite(m[key]) and m[key]>=0
        assert m['evaluate_seconds_mean']>0
        assert math.isclose(m['evaluate_seconds_total']/20,m['evaluate_seconds_mean'],rel_tol=1e-14)
    assert x['rhs_norm2']==y['rhs_norm2'] and x['surface_norm2']==y['surface_norm2']
    rows.append(dict(rank=rank,tile=a['mask'],first_row=x['first_row'],owned_rows=x['owned_rows'],
        staged_evaluate_seconds_mean=x['evaluate_seconds_mean'],inline_evaluate_seconds_mean=y['evaluate_seconds_mean'],
        inline_over_staged=y['evaluate_seconds_mean']/x['evaluate_seconds_mean'],
        staged_rhs_norm2=x['rhs_norm2'],inline_rhs_norm2=y['rhs_norm2'],
        staged_surface_norm2=x['surface_norm2'],inline_surface_norm2=y['surface_norm2'],
        staged_repeat_error=x['repeat_error'],inline_repeat_error=y['repeat_error']))
with (root/'benchmark-ranks.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
means={variant:math.fsum(row[variant+'_evaluate_seconds_mean'] for row in rows)/12 for variant in ['staged','inline']}
for variant,report in [('staged',staged),('inline',inline)]:
    assert means[variant]==report['timing']['evaluate_seconds_mean_rank_mean']
summary=dict(schema=1,scope='per-rank repeated Evaluate microbenchmark; no inter-rank start barrier',
    n=16,ranks=12,total_surface_rows=1536,owned_rows_per_rank=128,sampled_fields=25,
    warmup_evaluations=3,timed_repeats=20,variant_order=['staged','inline'],
    staged_evaluate_seconds_rank_mean=means['staged'],inline_evaluate_seconds_rank_mean=means['inline'],
    ratio_of_rank_means=means['inline']/means['staged'],
    paired_rank_ratio_min=min(row['inline_over_staged'] for row in rows),
    paired_rank_ratio_max=max(row['inline_over_staged'] for row in rows),
    per_rank_rhs_and_surface_norms_exactly_equal=True,cross_variant_full_array_comparison_performed=False,
    repeats_within_each_variant_unchanged=True,staged_jet_bytes_per_rank=332800,
    source_manifest_sha256=staged['source_manifest_sha256'],
    staged_binary_sha256=staged['ranks'][0]['result']['binary_sha256'],
    inline_binary_sha256=inline['ranks'][0]['result']['binary_sha256'],
    cmake_cache_sha256=staged['ranks'][0]['result']['cache_sha256'],
    limitations=['Output norms agree exactly as parsed floating-point values; equal norms do not prove equal full arrays.',
        'Each variant separately verifies unchanged full outputs across its repeats; this is distinct from cross-variant equality.',
        'Three warmups, state upload and setup are outside the timed interval; explicit Kokkos fences bracket timed Evaluate calls.',
        'The timed method includes its per-call host summary reduction, but no MPI exchange or coupled evolution.',
        'Staged precedes inline, with no inter-rank timing-start barrier; shared GPU resources can affect overlap.',
        'This ratio is not a whole-node throughput, allocation wall-time, or coupled evolution speedup claim.'])
(root/'benchmark-comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(f"Validated 12 owned row partitions; ratio of rank means={summary['ratio_of_rank_means']:.8f}")
