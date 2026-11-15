#!/usr/bin/env python3
"""Plot selected-rank GPU attribution; these are not throughput measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('summary', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--previous', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(a.summary.read_text())
    ranks = ['0', '8', '22', '23']
    labels = ['Bulk RHS', 'Agreement forward', 'Agreement transpose',
              'Mapped transport/fill', 'Other GPU work']
    colors = ['#4c78a8', '#e45756', '#f2a541', '#72b7b2', '#b6b6b6']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained', sharey=True)
    for ax, mode in zip(axes, ['legacy', 'masked', 'agreement']):
        values = np.zeros((5, 4))
        for ri, rank in enumerate(ranks):
            record = data['modes'][mode][rank]
            for name, stats in record['gpu_groups'].items():
                group = (0 if name.endswith('::CalcRHS') else
                         1 if name == 'z4c_hyperboloidal::OverlapAgreementPlan::Evaluate' else
                         2 if name == 'z4c_hyperboloidal::OverlapAgreementPlan::AddRHS' else
                         3 if 'MappedExchange' in name or 'MappedHalo' in name else 4)
                values[group, ri] += stats['sum_seconds']/record['cycles']
        base = np.zeros(4)
        for row, label, color in zip(values, labels, colors):
            ax.bar(np.arange(4), row, bottom=base, label=label, color=color)
            base += row
        ax.set(xticks=np.arange(4), xticklabels=ranks, xlabel='Selected MPI rank', title=mode.capitalize())
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Summed GPU seconds per RK cycle')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('Selected-rank GPU attribution\nInstrumented device work, not end-to-end throughput', fontsize=12)
    fig.savefig(a.output/'profile-attribution.png', dpi=180)
    if a.previous:
        old = json.loads(a.previous.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
        for ax, kernel in zip(axes, ['Evaluate', 'AddRHS']):
            key = 'z4c_hyperboloidal::OverlapAgreementPlan::'+kernel
            for offset, dataset, label, color in [(-.18, old, 'Before', '#b97740'),
                                                   (.18, data, 'Field-parallel', '#367baa')]:
                values = []
                for rank in ['0', '22', '23']:
                    r = dataset['modes']['agreement'][rank]
                    values.append(r['gpu_groups'][key]['sum_seconds']/r['cycles']*1000)
                ax.bar(np.arange(3)+offset, values, width=.36, label=label, color=color)
            ax.set(xticks=np.arange(3), xticklabels=['0', '22', '23'], xlabel='Selected MPI rank',
                   ylabel='GPU milliseconds per RK cycle', title='Discrepancy' if kernel == 'Evaluate' else 'Transpose')
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
        axes[0].legend(frameon=False)
        fig.suptitle('Same agreement operator; changed thread assignment')
        fig.savefig(a.output/'kernel-comparison.png', dpi=180)
    files = [Path(__file__), a.summary] + ([a.previous] if a.previous else [])
    (a.output/'plot-provenance.json').write_text(json.dumps(
        {str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, indent=2)+'\n')


if __name__ == '__main__':
    main()
