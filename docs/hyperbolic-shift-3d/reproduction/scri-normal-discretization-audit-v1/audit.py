#!/usr/bin/env python3
"""Bounded derivative-matrix arithmetic; no RHS, state or evolution evaluation.

The exact uniform examples use rational polynomial differentiation. The mapped
example uses the preserved donor's weights() and its verified retained r array.
"""
import argparse
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_weights(nodes,target,derivative):
    nodes=[F(x)-F(target) for x in nodes]
    result=[]
    for i,node in enumerate(nodes):
        polynomial=[F(1)];denominator=F(1)
        for j,other in enumerate(nodes):
            if i==j:continue
            product=[F(0)]*(len(polynomial)+1)
            for k,value in enumerate(polynomial):
                product[k]-=other*value;product[k+1]+=value
            polynomial=product;denominator*=node-other
        result.append(math.factorial(derivative)*polynomial[derivative]/denominator)
    return result


def rational_record(nodes,coefficients):
    return dict(nodes_over_h=[str(F(x)) for x in nodes],
                coefficients_exact=[str(x) for x in coefficients],
                coefficients=[float(x) for x in coefficients])


def moment_check(nodes,coefficients,derivative,degree):
    for power in range(degree+1):
        value=sum((c*F(x)**power for c,x in zip(coefficients,nodes)),F(0))
        expected=F(math.factorial(derivative)) if power==derivative else F(0)
        assert value==expected,(power,value,expected)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError("Output must be new")
    binding=json.loads((HERE/"source-binding.json").read_text())
    manifest_path=HERE/"original-1d-manifest.json"
    assert sha(manifest_path)==binding["original_manifest_sha256"]
    manifest=json.loads(manifest_path.read_text())
    assert manifest["checkpoint_hashes"]["profiles.npz"]==binding["original_profiles_sha256"]
    for source in binding["sources"]:
        assert sha(HERE/source["snapshot"])==source["sha256"]
        if source["snapshot"].startswith("source/one-d/"):
            assert manifest["sources"][source["repository_path"]]==source["sha256"]
    grid=HERE/"original-n513-r.npy"
    assert sha(grid)==binding["original_n513_r_npy_sha256"]
    r=np.load(grid,allow_pickle=False)
    assert hashlib.sha256(np.asarray(r,dtype="<f8").tobytes()).hexdigest()==binding["r_little_endian_float64_sha256"]
    assert len(r)==513 and r[0]==0 and r[-1]==1 and np.all(np.diff(r)>0)
    spec=importlib.util.spec_from_file_location("preserved_donor_numerics",HERE/"source/one-d/numerics.py")
    numerical=importlib.util.module_from_spec(spec);spec.loader.exec_module(numerical)
    spec=importlib.util.spec_from_file_location("preserved_donor_grid",HERE/"source/one-d/radial_grid.py")
    grid_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(grid_module)
    mapped,jacobian=grid_module.mapped_grid(np.linspace(0,1,513),0.,"inner-exp-scri")
    assert np.array_equal(r,mapped) and jacobian[-1]==.5

    # Donor regular-center outer rows clamp seven-node stencils to the domain.
    uniform=list(range(-6,1));d=exact_weights(uniform,0,1);d2=exact_weights(uniform,0,2)
    support=list(range(-9,1));composed=[F(0)]*10
    for coefficient,target in zip(d,uniform):
        start=min(target-3,-6);nodes=list(range(start,start+7))
        row=exact_weights(nodes,target,1)
        for node,value in zip(nodes,row):composed[node+9]+=coefficient*value
    moment_check(uniform,d,1,6);moment_check(uniform,d2,2,6)
    moment_check(support,composed,2,6)
    assert d[-1]==F(49,20) and d2[-1]==F(203,45) and composed[-1]==F(16709,3600)

    # Existing 3D normal sampler: endpoint plus five cell-center samples.
    half=[F(-9,2),F(-7,2),F(-5,2),F(-3,2),F(-1,2),F(0)]
    halfd=exact_weights(half,0,1);halfd2=exact_weights(half,0,2)
    moment_check(half,halfd,1,5);moment_check(half,halfd2,2,5)
    assert halfd[-1]==F(1126,315) and halfd2[-1]==F(10640,1323)
    ghost_nodes=[F(-11,2),*half]
    ghosts=[exact_weights(ghost_nodes,target,0)[-1] for target in [F(1,2),F(3,2)]]
    assert ghosts==[F(1024,231),F(1024,33)]

    n=len(r);hs=.5/(n-1)
    mapped_d=numerical.weights(r[-7:],1.,1)
    mapped_d2=numerical.weights(r[-7:],1.,2)
    mapped_composed=np.zeros(n)
    indexes=[]
    for k,i in enumerate(range(n-7,n)):
        start=min(i-3,n-7);indexes.append(list(range(start,start+7)))
        row=numerical.weights(r[start:start+7],r[i],1)
        for j,value in enumerate(row):mapped_composed[start+j]+=mapped_d[k]*value
    assert np.all(mapped_composed[:-10]==0)
    assert abs(mapped_d[-1]*hs-2.108470871939627)<2e-11
    assert abs(mapped_composed[-1]*hs**2-3.315087865896544)<2e-10
    assert abs(mapped_d2[-1]*hs**2-3.239383877601109)<2e-10
    result=dict(scope="Derivative matrices only; no field arrays, RHS, eigenproblem or evolution evaluated.",
        source_binding_sha256=sha(HERE/"source-binding.json"),audit_sha256=sha(Path(__file__)),
        exact_controls=dict(polynomial_moments=True,expected_rational_diagonals=True,
                            composed_support_ten_nodes=True,ghost_lift_weights=True,
                            original_grid_reconstructed_bitwise=True),
        uniform_one_d=dict(endpoint_D1=rational_record(uniform,d),direct_D2=rational_record(uniform,d2),
                          composed_D1D1=rational_record(support,composed),
                          composed_minus_direct_diagonal=str(composed[-1]-d2[-1])),
        three_d=dict(endpoint_D1=rational_record(half,halfd),direct_D2=rational_record(half,halfd2),
                     algebraic_pole_diagonal_after_4_over_h=str(F(4)-halfd[-1]),
                     ghost_nodes_over_h=[str(x) for x in ghost_nodes],
                     first_second_ghost_boundary_weights_exact=[str(x) for x in ghosts],
                     first_second_ghost_boundary_weights=[float(x) for x in ghosts]),
        mapped_original_n513=dict(hs=hs,grid_stretch=binding["original_grid_config"]["grid_stretch"],
            map="inner-exp-scri; outer Jacobian J(1)=1/2",outer_nodes_r=r[-10:].tolist(),
            interior_distances_over_hs=((1-r[-7:-1])/hs).tolist(),
            endpoint_D1_times_hs=(mapped_d*hs).tolist(),
            composed_D1D1_times_hs2=(mapped_composed[-10:]*hs**2).tolist(),
            direct_D2_times_hs2=(mapped_d2*hs**2).tolist(),
            first_derivative_row_indexes_zero_based=indexes,
            caveat="These are the donor's coordinate derivative matrices. Reference-variable transformations and nonlinear pole coefficients add further terms in the full RHS; this is not its full Jacobian."))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(exact_controls=result["exact_controls"],
        mapped_D1_diagonal= mapped_d[-1]*hs,
        mapped_composed_diagonal=mapped_composed[-1]*hs**2,
        mapped_direct_D2_diagonal=mapped_d2[-1]*hs**2),indent=2))


if __name__=="__main__":main()
