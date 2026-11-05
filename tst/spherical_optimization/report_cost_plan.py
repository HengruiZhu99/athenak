#!/usr/bin/env python3
"""Render measured cost tables from analyze_cost_plan's immutable JSON."""
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    d=json.loads(a.comparison.read_text());rows=d['summary']
    lines=['# Controlled local accuracy–cost result','',
      'The final benchmark compares the conservative guard, a qualified CFL .55',
      'step at penalty .6510416666660156, and penalty .5 with CFL .68. All cases',
      'use the same shared continuum formulation, central 20M scale, Release binary,',
      'execution-source hashes and diagnostic cadence. Only grid size, penalty and',
      'explicit step policy differ. See [COST-PROTOCOL.md](COST-PROTOCOL.md) for the',
      'targets declared before timing and [PENALTY-TRADEOFF.md](PENALTY-TRADEOFF.md)',
      'for finite-duration pulse and timestep qualification.','',
      'Each of nine cases ran serially three times in rotated order, with library',
      'thread counts fixed to one. No other laboratory evolution or spectral/plot',
      'workload ran during this batch; unrelated OS activity was not excluded.',
      'All three final states agreed bitwise in each of the nine cases.',
      'Timings below include stepping diagnostics and output. Setup and full',
      'subprocess times are retained separately in the JSON.','',
      '## Measured 10M times and accuracy','',
      '| Policy | N | CPU stepping seconds, median [min,max] | Exterior E | Clock magnitude | Max D defect |',
      '|---|---:|---:|---:|---:|---:|']
    if d.get('hardware'):
        h=d['hardware'];index=lines.index('## Measured 10M times and accuracy');lines[index:index]=[f"Measured host: {h['machdep.cpu.brand_string']}, {int(h['hw.memsize'])/2**30:g} GiB memory, {h['hw.ncpu']} logical CPUs. No CPU affinity was imposed.",'']
    for r in rows:
        if r['completed']!=r['repeats']:
            lines.append(f"| {r['family']} | {r['n']} | incomplete: {r['completed']}/{r['repeats']} | — | — | — |");continue
        t=r['stepping_seconds'];lines.append(f"| {r['family']} | {r['n']} | {t['median']:.4f} [{t['min']:.4f},{t['max']:.4f}] | {r['screening_indicator']:.7g} | {abs(r['clock']):.7g} | {r['D_defect_max']:.7g} |")
    lines+=['','E here is the larger of PCHIP and linear-interpolated maximum dimensionless',
      'physical H/M/Theta/Z RMS over 301 logarithmic reference R/M25–100 samples.',
      'Both interpolation estimates must pass a target. It is a constraint indicator,',
      'not a certified solution-error bound. Clock and D are separate fixed laboratory',
      'coordinate quantities (M=1/3); D uses the maximum of three directional probes.',
      'The coarse N128 boundary residuals exclude it from every boundary-aware box.','',
      '## Fixed-target selections','',
      '| Screen | Policy | Cheapest measured N | Median seconds | Conservative/policy cost ratio |',
      '|---|---|---:|---:|---:|']
    for key in ['exterior_targets','boundary_aware_targets']:
        for t in d[key]:
            label=f"exterior E≤{t['E']:g}" if key=='exterior_targets' else f"{t['name']}: E≤{t['E']:g}, clock≤{t['clock']:g}, D≤{t['D']:g}"
            for family,v in t['selected'].items():
                ratio=t['baseline_over_family_cost_ratio'][family]
                lines.append(f"| {label} | {family} | {v['n'] if v else 'none'} | {format(v['stepping_seconds']['median'],'.4f') if v else '—'} | {format(ratio,'.3f')+'×' if ratio is not None else 'no finite ratio'} |")
    lines+=['','These are cheapest measured cases within this small family; there is no',
      'grid-count extrapolation or global-optimum claim. Tiny differences between',
      'near-identical exterior indicators should not be interpreted as significant',
      'ranking of penalty accuracy. The meaningful additional benefit is the larger',
      'qualified step at comparable frozen-spectrum margin. The selected boundary',
      'boxes require more points than the loose exterior-only screen.','',
      '## Step and diagnostic accounting','',
      '| Policy | N | Steps | RHS calls | dt/M | min dr/M | Steps/M | Max accepted characteristic CFL |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['completed']!=r['repeats']:continue
        v=lambda key:r[key]['median']
        lines.append(f"| {r['family']} | {r['n']} | {v('steps'):.0f} | {v('rhs_calls'):.0f} | {v('dt_over_M'):.8g} | {v('min_dr_over_M'):.8g} | {v('steps_per_M'):.1f} | {v('C_char_max_accepted'):.8g} |")
    speeds=[r['worst_evaluated_speed']['max'] for r in rows if r['completed']==r['repeats']]
    lines+=['',f"Initial maximum coordinate speed is 2.414213562373095; the largest evaluated speed in this batch is {max(speeds):.12g}.",
      'The conservative policy is limited by its scri guard; optimized cases use',
      'the explicit experimental fixed step. Final shortened steps and all report',
      'RHS calls count in these totals. No change of time normalization is counted',
      'as a speedup.','',
      '## Scope and reproduction','',
      'The historical matched areal baseline in [ACCURACY-COST.md](ACCURACY-COST.md)',
      'misses all three exterior targets. No finite cost ratio against that baseline',
      'is inferred. The ratios above isolate step/penalty optimization within the',
      'final shared continuum family. Outer damping .25 improves boundary regularity',
      'at the price of larger exterior error on a fixed grid; that tradeoff remains',
      'in [OUTER-LAPSE.md](OUTER-LAPSE.md).','',
      'The working CFL .68 has matched pulse controls through 40M at N128/256 and',
      'unperturbed fine-grid accuracy through 10M at N512. A late gauge-response',
      'upturn remains resolved and preserved. Neither local completion nor these',
      'CPU speedups establish all-time, boosted or 3D stability, or GPU-hour gains.',
      'The separate [3D transfer gate](HANDOFF-AUDIT.md) remains mandatory.','',
      'Run from the repository root, with fresh output directories:','',
      '```sh',
      'build-spherical/python/bin/python tst/spherical_optimization/run_cost_plan.py \\',
      '  --library build-spherical-v26/libcurrent_spherical.dylib \\',
      '  --specification docs/spherical-optimization/evidence/cost-specification-v28.json \\',
      '  --output fresh-serial-cost --repeats 3',
      'build-spherical/python/bin/python tst/spherical_optimization/analyze_cost_plan.py \\',
      '  --library build-spherical-v26/libcurrent_spherical.dylib \\',
      '  --runs fresh-serial-cost --output fresh-cost-analysis',
      'build-spherical/python/bin/python tst/spherical_optimization/report_cost_plan.py \\',
      '  --comparison fresh-cost-analysis/comparison.json --output fresh-cost-report.md',
      '```','',
      'The archived results are `evidence/serial-cost-v28` and',
      '`evidence/serial-cost-final-v28`. Source/dependency identities and the',
      'complete run index are in `evidence/penalty-controls-source-v28`. The local',
      'v26 binary and build/test records remain in `evidence/outer-lapse-build-v26`.',
      'Use the rebuilt platform library (`.so` on Linux) when reproducing elsewhere;',
      'runtime numbers are hardware-specific.','',
      '![Separate exterior, clock and D accuracy–cost curves](evidence/serial-cost-final-v28/accuracy-cost.png)','']
    a.output.write_text('\n'.join(lines))
if __name__=='__main__':main()
