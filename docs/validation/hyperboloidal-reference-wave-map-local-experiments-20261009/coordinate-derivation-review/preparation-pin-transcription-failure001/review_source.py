"""Hash-bound read-only mathematical/source review, not a numerical gate."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "boundary/einstein-coordinate-gauge-held-20261009"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


capture = SOURCE / "source-only-preparation.json"
assert sha(capture) == "5f6f91568d30040bb99233b8ba2fead883db5bce3e478630b18df595e491d60c"
data = json.loads(capture.read_text())
pins = []
for entry in data["documents"] + data["inputs"]:
    path = SOURCE / entry["path"]
    assert sha(path) == entry["sha256"]
    assert path.stat().st_size == entry["bytes"]
    pins.append({"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size})
assert sha(SOURCE / "DERIVATION.md") == "126667e9e0bf61eb8f245cf9018ace90b2c35d6631d58f48f5fe40c23b07b83d"
result = {
    "status": "PASS_read_only_source_math_review", "review_source_sha256": sha(__file__),
    "preparation_sha256": sha(capture), "reviewed_pins": pins,
    "conventions": {"active_metric_variation": "delta g4=Lie_xi g4",
                    "physical_extrinsic_curvature": "Kij=-Lie_n gammaij/2",
                    "coordinate_time_generator": "T is contravariant xi^t",
                    "external_conformal_factor": "Omega is fixed; psi=X.gradOmega/Omega",
                    "scope": "positive finite Omega, stationary Minkowski layer reference"},
    "findings": [
        "Multiplying the physical Lie variation by fixedOmega^2 gives delta barg4=Lie_xi barg4-2psi barg4. ADM lapse/shift signs follow the contravariant time generator and negative-K convention.",
        "The vacuum normal deformation with N=aT,Y=X+betaT reproduces hphys and kij. Expanding Lie_(betaT)K and Hessian(aT) cancels T times the stationary ADM K equation; both beta^k K_ki T_j terms and both a_i T_j terms are retained.",
        "The trace variation is gamma^-1:k-K^ij h_ij, including inverse-metric variation. Inverting ToPhysicalADM yields deltaA=Omega chi[kij-K hphys/3-gamma k/3]+(delta chi/chi)Aref, with the required live reference A trace tangent.",
        "For detgtilde=1 and its exact tangent, deltaLambda=partial_j(ginv delta g ginv) is equivalent to varying the contracted Christoffel symbol; all reference coefficient derivatives must be retained. This gives deltaZ=0 without an output projection.",
        "The stationary coordinate-kinematic derivative uses the lift at Tdot,Xdot for geometric fields. Differentiating the two ADM gauge variations gives the stated Tddot/Xddot formulas, including -Xdot.grad(log physical lapse).",
        "The physical-P lapse Jacobian matches InteriorLayerGauge: advection, -nu deltaalpha, -f2 deltaP/Omega, -2W xi alpha deltaalpha/Omega and -W(alpha deltabeta+beta deltaalpha).gradOmega/Omega.",
        "The original shift Jacobian has mu alpha^2 chi deltaLambda and gradient log-ratio variations. All live coefficient variations multiply vanishing reference deviations, hence no extra first variation.",
        "The spatialnorm pole uses n=-gradOmega/|gradOmega|, eta6,C2/3 and deltaG from inverse-metric/chi variations. It must be assembled once; GenericGauge adds beta pole after the production interior assembly. No Ghat division belongs in the exact W0 branch.",
        "The flat-core lift is deltaP=-Delta tau, deltaA=-4tau_rhorho STF(xx), deltaLambda=(8/3)x(5zeta_rho+2rho zeta_rhorho). Falpha=-3deltaP,Fbeta=3deltaLambda/8 give v_t=18tau_rho+12rho tau_rhorho and w_t=2v_rho+5zeta_rho+2rho zeta_rhorho.",
        "The stated jet budget is consistent: input metric/lapse/shift second jets require configuration/reference third derivatives, input A/P first jets require physicalK second derivatives, and L third derivatives require Omega fourth derivatives. Kinematic output constraints additionally require velocity third derivatives.",
        "Finite point values cannot independently differentiate the actual RHS. Future source-jet or constraint-rate gates must specify higher jets or convergent FD; the recipe correctly keeps that issue separate."
    ],
    "future_gates_retained": [
        "independent four-metric/ADM lift and general/stationary K comparison",
        "stored/physical roundtrip, det/A normals and physical H/M/Z/Theta before projection",
        "complete consumed old jets plus independent new high-precision jets including exp-flat tails",
        "generic-dual actual gauge, double directional differences, omitted-pole and wrapper controls",
        "independent core polynomial fields including constants/origin and exact branches",
        "raw22 actual geometry action versus kinematic lift at held-out finiteOmega points"
    ],
    "corrections": [], "scientific_execution": False,
    "operator_spectra_propagation_or_boundary_admitted": False,
    "limitations": "No exact-scri falloff, coordinate boundary condition, CPBC, invariant Einstein projected polynomial space, continuum energy or stability follows. Later single-BH wormhole-to-trumpet transition with the Minkowski reference remains unresolved."
}
(HERE / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
copies = HERE / "reviewed-documents"
copies.mkdir(exist_ok=False)
for name in ("DERIVATION.md", "HELD-RECIPE.md", "source-only-preparation.json"):
    shutil.copyfile(SOURCE / name, copies / name)
print(json.dumps({"status": result["status"], "receipt_sha256": sha(HERE / "receipt.json"), "pins": len(pins)}))
