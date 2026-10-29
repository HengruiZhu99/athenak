#!/usr/bin/env python3
"""Summarize retained angular-operator evidence; no evolution or remote access."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from prototype import Plan

HERE=Path(__file__).resolve().parent
NS=(6,8,12,16,24,32)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=HERE/'cases')
    parser.add_argument('--output',type=Path,default=HERE)
    parser.add_argument('--copy-cases',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cases={};bindings={}
    for degree in (5,7):
        for n in NS:
            name=f'n{n}-p{degree}';path=args.input/name/'report.json'
            report=json.loads(path.read_text());cases[n,degree]=report
            if report['prototype_sha256']!=sha(HERE/'prototype.py'):raise ValueError('prototype source hash mismatch')
            native=report['native_geometry'];checks=report['checks']
            if not(native['labels_exact'] and native['coordinate_max']<2e-14 and native['midpoint_area_relative_max']<1e-12):
                raise ValueError('native geometry mismatch')
            if not(checks['constant_B_max']==checks['constant_Q_max']==0 and checks['energy_identity_relative_max']<1e-12
                   and checks['weighted_adjoint_relative_max']<1e-11 and checks['cube_rotation_commutator_relative_max']<1e-11):
                raise ValueError('algebraic operator gate failed')
            bindings[name]=dict(report_sha256=sha(path),prototype_sha256=report['prototype_sha256'])
            if args.copy_cases:
                dest=out/'cases'/name;dest.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,dest/'report.json')
                if n==6:shutil.copy2(path.with_name('plan.npz'),dest/'plan.npz')
    table=[];orders=[]
    for degree in (5,7):
        for n in NS:
            r=cases[n,degree];p=r['plan'];c=r['checks']
            table.append(dict(degree=degree,n=n,neighbors=p['neighbor_max'],incoming=p['incoming_max'],
                              condition=p['condition_max'],row_l1=p['coefficient_l1_max'],
                              spectral_upper=c['normalized_spectral_upper_bound'],
                              power_rayleigh=c['normalized_power_estimate']['rayleigh'],
                              power_residual=c['normalized_power_estimate']['residual'],
                              isolated_rk4_dt_epsilon_n=c['rk4_isolated_dt_epsilon_over_h_upper']))
        for previous,n in zip(NS,NS[1:]):
            a=cases[previous,degree]['truncation'];b=cases[n,degree]['truncation']
            for field in a:
                for region in a[field]:
                    for norm in ('max','l2'):
                        old=a[field][region][norm];new=b[field][region][norm]
                        orders.append(dict(degree=degree,coarse=previous,fine=n,field=field,region=region,norm=norm,
                                           old=old,new=new,order=float(np.log(old/new)/np.log(n/previous))))
    for filename,rows in [('summary.csv',table),('orders.csv',orders)]:
        with (out/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dense={}
    for degree in (5,7):
        source=args.input/f'n6-p{degree}'/'plan.npz';saved=np.load(source)
        plan=Plan(6,degree,*[saved[k] for k in ('points','labels','mass','starts','rows','indices','coefficients')],{})
        root=np.sqrt(plan.mass);matrix=-root[:,None]*plan.apply_q(np.eye(len(root)))/root[None,:]/6
        eigen=np.linalg.eigvalsh((matrix+matrix.T)/2)
        dense[degree]=dict(plan_sha256=sha(source),symmetric_residual_max=float(np.max(np.abs(matrix-matrix.T))),
                           symmetric_part_eigen_min=float(eigen.min()),symmetric_part_eigen_max=float(eigen.max()),
                           upper_bound=plan.spectral_upper_bound(),
                           row_l2_max=float(np.max(np.sqrt(np.add.reduceat(plan.coefficients**2,plan.starts[:-1])))))
        if eigen.min()<-1e-12 or eigen.max()>plan.spectral_upper_bound():raise ValueError('dense spectrum gate failed')
    assessment={}
    for degree in (5,7):
        last=[r['order'] for r in orders if r['degree']==degree and r['fine']==32 and r['norm']=='max']
        penultimate=[r['order'] for r in orders if r['degree']==degree and r['fine']==24 and r['norm']=='max']
        assessment[degree]=dict(final_interval_maxnorm_order_min=min(last),penultimate_interval_maxnorm_order_min=min(penultimate),
                                bounded_study_only=True,formal_smooth_order=degree,
                                formal_surface_only_two_radial_derivative_order=degree-2)
    write(out/'assessment.json',dict(scope='standalone operator; production integration, closure feedback and nonlinear stability untested',
                                    recommendation='degree7 qualifies for a controlled opt-in native candidate, not a stable-evolution admission',
                                    cases=bindings,degree_assessment=assessment,dense_n6_checks=dense,
                                    mode_scope='Only frozen-volume V2 surface modes0 and2 are independently resolved; no full-coupled mode claim',
                                    source_files={p.name:sha(p) for p in (HERE/'prototype.py',HERE/'native_geometry.cpp',Path(__file__))}))
    fig,axes=plt.subplots(2,2,figsize=(10.6,7.6),layout='constrained')
    colors={5:'#b45b22',7:'#235a9f'}
    for degree in (5,7):
        for field,style in [('vector_x','-'),('tensor_01','--'),('harmonic_l6',':')]:
            axes[0,0].loglog(NS,[cases[n,degree]['truncation'][field]['all']['max'] for n in NS],
                            marker='o',color=colors[degree],linestyle=style,label=f'p={degree}, {field}')
        for region,style in [('interior','-'),('edge','--'),('corner',':')]:
            axes[0,1].loglog(NS,[cases[n,degree]['truncation']['smooth_exp'][region]['max'] for n in NS],
                            marker='o',color=colors[degree],linestyle=style,label=f'p={degree}, {region}')
        axes[1,0].plot(NS,[cases[n,degree]['checks']['normalized_spectral_upper_bound'] for n in NS],
                       'o-',color=colors[degree],label=f'p={degree}, certified bound')
        axes[1,0].plot(NS,[cases[n,degree]['checks']['normalized_power_estimate']['rayleigh'] for n in NS],
                       's--',color=colors[degree],label=f'p={degree}, power Rayleigh')
        for index,offset in [(0,-.16),(2,.16)]:
            value=cases[6,degree]['surface_modes']['measurements'][index]['damping_rayleigh']
            axes[1,1].bar(degree+offset,value,width=.28,color=colors[degree],alpha=1 if index==0 else .55,
                          label=f'p={degree}, mode{index}')
    axes[0,0].set(title='Smooth spherical scalar/vector/tensor data',ylabel=r'$\|Q u\|_\infty$, $\epsilon=1$')
    axes[0,1].set(title='Nonpolynomial smooth field by face region',ylabel=r'$\|Q u\|_\infty$, $\epsilon=1$')
    axes[1,0].set(title='Dimensionless spectral scale',ylabel=r'$\lambda(M^{-1/2} B^T M B M^{-1/2})$')
    axes[1,1].set(title='Resolved n=6 V2 surface modes only',ylabel=r'$-\mathrm{Re}(v^*M Qv)/(v^*Mv)$',xlabel='Polynomial degree',xticks=[5,7])
    for ax in axes.flat:ax.grid(alpha=.22);ax.legend(fontsize=7)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:
        ax.set_xlabel('Cells per patch axis n');ax.set_xticks(NS,labels=NS)
        ax.xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle('Angular dissipation prototype — operator consistency and energy, not evolution stability',fontsize=12)
    fig.savefig(out/'operator-admission.png',dpi=180);fig.savefig(out/'operator-admission.pdf');plt.close(fig)
    print(json.dumps(dict(assessment=assessment,dense=dense)),flush=True)

if __name__=='__main__':main()
