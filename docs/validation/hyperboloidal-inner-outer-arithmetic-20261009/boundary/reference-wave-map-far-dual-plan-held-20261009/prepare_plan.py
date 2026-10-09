#!/usr/bin/env python3
"""Metadata-only one-shot plan preparation; no scientific imports or arithmetic."""
import hashlib
import itertools
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent.parent
OUTER = HERE.parent / "reference-wave-map-outer-arithmetic-source001-held-20261009"
REVIEW = BASE / "continuum/reference-wave-map-outer-arithmetic-independent-review-20261009"
FAILURE = HERE.parent / "inner-source003-failure-saved-readback-20261009"

def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def pin(path):
    path = pathlib.Path(path).absolute()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}

def write(path, value):
    path = pathlib.Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

def read(path):
    return json.loads(pathlib.Path(path).read_text())

def check(row):
    actual = pin(row["path"])
    assert actual == {k: row[k] for k in ("path", "sha256", "bytes")}, row["path"]

def main():
    start = time.time()
    assert not (HERE / "source-index.json").exists(), "fresh one-shot destination required"
    assert sha(OUTER / "source-index.json") == "9da21a96d243441c95f276d1c076d3a4e561fc20921858be33f911cb3464f27e"
    assert sha(REVIEW / "index.json") == "d3bae95086feedf0e2010af7705ecae9d3e1cef6f44f34ee4e1343ef5438e242"
    assert sha(FAILURE / "index.json") == "d08c71e907d29b16d7cb28642f1ce015a6987c68bec78327f9bd43c8e4660fb6"
    protected = {}
    for row in read(OUTER / "input-pins.json") + read(OUTER / "source-index.json")["files"]:
        protected[row["path"]] = {k: row[k] for k in ("path", "sha256", "bytes")}
    extras = [OUTER / "source-index.json", REVIEW / "index.json", REVIEW / "receipt.json",
              REVIEW / "REVIEW.md", FAILURE / "index.json", FAILURE / "attempt001/summary.json",
              FAILURE / "PENCIL-ONLY-OUTER-CORRECTION.md", pathlib.Path(sys.executable)]
    for path in extras:
        row = pin(path)
        protected[row["path"]] = row
    protected_rows = sorted(protected.values(), key=lambda row: row["path"])
    for row in protected_rows:
        check(row)
    source_before = [pin(HERE / name) for name in ("prepare_plan.py", "PLAN.md", "ORACLES.md")]
    e = ["1", "-1/2", "1/4"]
    ze = ["0", "0", "0"]
    beta = ["1/16", "-1/32", "1/64"]
    beta_d = [["1/32", "-1/64", "1/128"],
              ["-1/64", "1/128", "-1/256"],
              ["1/128", "-1/256", "1/512"]]
    lam = ["1/64", "-1/128", "1/256"]
    def seed(name, **changes):
        result = {"id": name, "xi_alpha": "0", "xi_chi": "0",
                  "zeta_alpha": ze, "zeta_chi": ze,
                  "beta_value_dot": ze, "beta_d_dot": [["0"]*3 for unused in range(3)],
                  "Lambda_value_dot": ze, "P_value_dot": "0", "Theta_value_dot": "0",
                  "metric_value_dot": "0", "unconsumed_jets_dot": "0"}
        result.update(changes)
        return result
    seeds = [
        seed("zero"), seed("alpha-relative", xi_alpha="1"),
        seed("chi-relative", xi_chi="1"),
        seed("joint-relative", xi_alpha="1", xi_chi="1"),
        seed("A0-balanced", xi_alpha="1", xi_chi="-2"),
        seed("alpha-chi-balanced", xi_alpha="1", xi_chi="-1"),
        seed("alpha-gradient-only", zeta_alpha=e),
        seed("chi-gradient-only", zeta_chi=e),
        seed("A0-balanced-plus-gradients", xi_alpha="1", xi_chi="-2",
             zeta_alpha=e, zeta_chi=["-2", "1", "-1/2"]),
        seed("beta-value", beta_value_dot=beta),
        seed("beta-derivative", beta_d_dot=beta_d),
        seed("Lambda-value", Lambda_value_dot=lam),
        seed("physical-P", P_value_dot="1/128"),
        seed("metric-STF", metric_value_dot="g E+E g; E=diag(1,-1,0)/32"),
        seed("Theta-only", Theta_value_dot="1"),
        seed("all-used-mixed", xi_alpha="1", xi_chi="-2", zeta_alpha=e,
             zeta_chi=e, beta_value_dot=beta, beta_d_dot=beta_d,
             Lambda_value_dot=lam, P_value_dot="1/128", Theta_value_dot="1",
             metric_value_dot="g E+E g; E=diag(1,-1,0)/32"),
        seed("unconsumed-jet-only", unconsumed_jets_dot={
            "A_value_dot[i][j]": "e_i e_j/64",
            "metric_d_dot[k][i][j]": "e_k e_i e_j/128",
            "metric_dd_dot[k][l][i][j]": "e_k e_l e_i e_j/256"})]
    variants = ["alpha-gradient-only", "chi-gradient-only",
                "A0-balanced-plus-gradients", "all-used-mixed"]
    families = [{"name": "collapsed", "index": 10, "alpha": "1e-150", "chi": "1e-150"},
                {"name": "small-alpha-large-chi", "index": 11, "alpha": "1e-150", "chi": "1e60"},
                {"name": "large-alpha-small-chi", "index": 12, "alpha": "1e60", "chi": "1e-150"},
                {"name": "chi-gradient-contrast", "index": 13, "alpha": "1e100", "chi": "1e-150"}]
    bases = []
    for a, radius, direction, family in itertools.product(
            [".5", "2"], [".025", ".1", ".5", ".65", ".84", ".95", ".995"],
            [0, 1], families):
        bases.append({"id": "B%03d" % len(bases), "a": a, "radius": radius,
                      "direction": direction, "family": family["name"], "family_index": family["index"]})
    closed_seeds = [["0", "0", "0"], ["1", "0", "0"], ["0", "1", "0"],
                    ["1", "1", "0"], ["1", "-2", "0"], ["0", "0", "1"]]
    registry = {
        "source_only": True, "execution_admitted": False,
        "number_parsing": "strings denote exact supplied binary64 decimal/hex construction; no targets evaluated here",
        "fixed_reference": {"S": "1", "geometry_r0": ".05", "geometry_r1": ".95",
                            "gauge_W_r0": ".45", "gauge_W_r1": ".85", "G0_provenance_only": ".375",
                            "directions": [["1", "0", "0"], [".36", "-.48", ".8"]],
                            "reference_and_xyz_field_dual": "all zero"},
        "State_constructor": {"path": str(OUTER / "probe.cpp"), "function": "State(p,family)",
                              "unchanged_base_fields": True, "family_definitions": families},
        "base_order": "a,radius,direction,family; listed order, then seeds in listed order",
        "bases": bases, "e": e, "seeds": seeds,
        "zero_default": "every field jet tangent not explicitly assigned is zero; do not call Consistent or alter raw seed afterward",
        "field_dual_formula": {"alpha_dot": "xi_alpha*alpha", "chi_dot": "xi_chi*chi",
            "alpha_d_dot[j]": "xi_alpha*alpha_d[j]+alpha*zeta_alpha[j]",
            "chi_d_dot[j]": "xi_chi*chi_d[j]+chi*zeta_chi[j]"},
        "tensor_index_conventions": "beta_d[j][i]=partial_j beta^i; symmetric metric rows i,j, derivative rows k,l",
        "local_seed_scope": "independent local field jets; no implicit spatial integrability/Einstein/algebraic tangent claim",
        "zero_primal_gradient_variants": {"base_order": "same112", "before_seeding": "alpha_d=chi_d=0 exactly",
                                          "seed_ids": variants},
        "closed_defaults": {"g_and_gh": "I", "metric_spatial_jets": "0", "beta_and_betahat": "0",
            "Lambda_and_Lambdahat": "0", "P_and_Phat": "0", "Theta_and_Thetahat": "0",
            "Omega": "1", "connection": "all scaled entries0; valid=true supplied directly",
            "reference_alpha_h_and_chi_y": "1", "unstated_gradients_and_duals": "0",
            "reference_alpha_gradient_adapter": "p.dalpha=p.state.alpha.d, fixed field dual0",
            "reference_beta_adapter": "p.beta=p.state.beta.value", "reference_P_adapter": "p.k_physical=p.state.trace.value"},
        "closed_seed_order_xi_alpha_xi_chi_xi_gradient": closed_seeds,
        "closed_contexts": [
            {"id": "tensor-pole", "alpha": "2^-300", "chi": "2^601", "Omega_d": ["1/2", "0", "0"],
             "target_component": "pole_beta_x and assembled_beta_x", "primal_target": "1",
             "dual_target": "4*xi_alpha+2*xi_chi", "legacy_zero_seed_primal_prediction": "0"},
            {"id": "chi-flux", "alpha": "2^300", "chi": "2^-600", "Omega_d": ze,
             "chi_d": ["-chi", "0", "0"], "reference_chi_d": ["1", "0", "0"],
             "chi_d_dot_x": "-chi*xi_chi+chi*xi_gradient", "target_component": "regular_beta_x and assembled_beta_x",
             "primal_target": "-1", "dual_target": "-xi_alpha-xi_chi/2+xi_gradient/2", "legacy_zero_seed_primal_prediction": "0"},
            {"id": "alpha-flux", "alpha": "2^-300", "chi": "2^600", "Omega_d": ze,
             "alpha_d": ["alpha", "0", "0"], "reference_alpha_d": ["2", "0", "0"],
             "alpha_d_dot_x": "alpha*(xi_alpha+xi_gradient)", "target_component": "regular_beta_x and assembled_beta_x",
             "primal_target": "1", "dual_target": "-2*xi_alpha-xi_chi-xi_gradient", "legacy_zero_seed_primal_prediction": "2^301"}],
        "negative_controls": "three zero-seed legacy exports reused; no additional helper calls",
        "FD": {"base_filter": {"a": ".5", "direction": 1, "radius": [".025", ".5", ".84", ".995"],
                               "families": [row["name"] for row in families]},
               "seed": "alpha-relative", "path": "field-linear u(s)=u0+s*udot, full registered jet",
               "steps": ["1e-3", "5e-4", "2.5e-4", "1.25e-4", "6.25e-5"], "sides": ["plus", "minus"]},
        "counts": {"bases": 112, "base_dual_rows": 1904, "zero_gradient_variant_rows": 448,
                   "physical_reference_dual_rows": 2352, "closed_positive_dual_rows": 18,
                   "positive_dual_rows": 2370, "legacy_negative_reuse_rows": 3, "records": 2373,
                   "FD_representatives": 16, "FD_side_evaluations": 160, "total_helper_evaluations": 4900}}
    assert len(bases) == 112 and len(seeds) == 17 and len(variants) == 4
    assert len({row["id"] for row in seeds}) == 17
    write(HERE / "CASE-REGISTRY.json", registry)
    write(HERE / "input-pins.json", protected_rows)
    write(HERE / "recipe.json", {
        "status": "source-only held plan; no implementation or execution admitted",
        "scope": "direct outer001 RWM field-dual supplement, independent of compound inner acceptance",
        "source_identity": {"outer_source_index_sha256": sha(OUTER / "source-index.json"),
                            "helper_sha256": sha(OUTER / "inputs/reference_wave_map.hpp"),
                            "LegacyGauge_sha256": sha(OUTER / "inputs/reference_wave_map_legacy.hpp"),
                            "Product_traits_sha256": sha(OUTER / "inputs/arithmetic_traits.hpp"),
                            "main_registry_probe_sha256": sha(OUTER / "probe.cpp"),
                            "main_oracle_sha256": sha(OUTER / "oracle.py")},
        "original_main_registry_and_thresholds_changed": False,
        "prior_source003_overall_FAIL_preserved": True,
        "required_future_modes": ["direct-complete-field-dual", "closed-exact-flux", "relative-alpha-FD"],
        "future_precision_decimal_digits": [480, 560],
        "gates": {"MP_precision_scaled": "1e-220", "entrywise_primal_and_dual_scaled": "2e-10",
                  "closed_nonzero_normal_relative": "2e-10", "closed_exactzero_absolute": "2e-10",
                  "FD_final_entrywise_scaled": "5e-7", "FD_convergence": "first>=2*final OR every level<=5e-9"},
        "error_scale": "max(1,abs(native entry),abs(target entry)), individually per primal/dual output",
        "future_source_gate": "fresh independently reviewed probe/oracle/runner and exact root local release required",
        "scientific_execution_admitted": False, "operator_admitted": False,
        "spectrum_admitted": False, "propagation_admitted": False,
        "counts": registry["counts"]})
    for row in protected_rows + source_before:
        check(row)
    write(HERE / "preparation-receipt.json", {"completed": True, "returncode": 0,
        "inputs_unchanged": True, "source_only": True, "protected_input_count": len(protected_rows),
        "source_before": source_before, "scientific_imports": False, "compiler_queries": False,
        "target_oracle_or_numerical_evaluation": False, "only_bookkeeping_counts_evaluated": True,
        "wall_seconds": time.time()-start, "counts": registry["counts"]})
    files = [pin(path) for path in sorted(HERE.iterdir()) if path.is_file()]
    write(HERE / "source-index.json", {"source_only": True, "execution_admitted": False,
        "files": files, "protected_input_count": len(protected_rows),
        "scope": "fixed far-field complete-dual plan only; no science implementation or execution"})
    print(json.dumps({"source_only": True, "completed": True, "source_index_sha256": sha(HERE / "source-index.json"),
                      "protected_input_count": len(protected_rows), "files": len(files), "counts": registry["counts"]}, sort_keys=True))

if __name__ == "__main__":
    main()
