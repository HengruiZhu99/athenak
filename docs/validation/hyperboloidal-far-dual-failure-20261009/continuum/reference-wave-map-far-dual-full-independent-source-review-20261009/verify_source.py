"""Standard-library hash/AST/registry readback; no oracle target evaluation."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAP = HERE / "captured"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    before = json.loads((HERE / "inputs-before-review.json").read_text())
    for item in before["files"]:
        if sha(item["path"]) != item["sha256"] or sha(HERE / item["captured"]) != item["sha256"]:
            raise RuntimeError("owner/capture drift: " + item["path"])
    protected = json.loads((CAP / "input-pins.json").read_text())
    for item in protected:
        if sha(item["path"]) != item["sha256"] or Path(item["path"]).stat().st_size != item["bytes"]:
            raise RuntimeError("protected origin drift: " + item["path"])
    recipe = json.loads((CAP / "recipe.json").read_text())
    registry = json.loads((CAP / "inputs/plan/CASE-REGISTRY.json").read_text())
    outer = Path(recipe["outer_source_index"]["path"]).parent
    plan = Path(recipe["approved_plan_index"]["path"]).parent
    for name in ["PLAN.md", "ORACLES.md", "CASE-REGISTRY.json", "recipe.json", "index.json"]:
        if (CAP / "inputs/plan" / name).read_bytes() != (plan / name).read_bytes():
            raise RuntimeError("approved plan copy changed: " + name)
    headers = ["reference_wave_map.hpp", "reference_wave_map_legacy.hpp", "arithmetic_traits.hpp", "dual_helpers.hpp", "nonlinear_values.hpp"]
    for name in headers:
        if (CAP / "inputs" / name).read_bytes() != (outer / "inputs" / name).read_bytes():
            raise RuntimeError("accepted outer input changed: " + name)
    original = (outer / "probe.cpp").read_text()
    def extraction(start, end):
        begin = original.index(start)
        return original[begin:original.index(end, begin)]
    if extraction("hyp::Z4cJet<double> State(", "template<class T>void EmitGauge(") != (CAP / "inputs/original_state.hpp").read_text():
        raise RuntimeError("original 112-state constructor extraction changed")
    if extraction("hyp::LayerPoint<D> CastPoint(", "Jet Direction(") != (CAP / "inputs/original_cast.hpp").read_text():
        raise RuntimeError("original reference cast extraction changed")
    history = outer.parent / "reference-wave-map-far-dual-source001-held-20261009"
    for name in ["probe.cpp", "oracle.py"]:
        if (CAP / name).read_bytes() != (history / name).read_bytes():
            raise RuntimeError("source002 changed scientific source001 bytes: " + name)
    for name in ["oracle.py", "run_once.py", "prepare_sources.py"]:
        ast.parse((CAP / name).read_text())
    expected_seed_names = ["zero", "alpha-relative", "chi-relative", "joint-relative", "A0-balanced", "alpha-chi-balanced", "alpha-gradient-only", "chi-gradient-only", "A0-balanced-plus-gradients", "beta-value", "beta-derivative", "Lambda-value", "physical-P", "metric-STF", "Theta-only", "all-used-mixed", "unconsumed-jet-only"]
    if [s["id"] for s in registry["seeds"]] != expected_seed_names:
        raise RuntimeError("seed names/order differ from independently read probe")
    expected_bases = [(a, r, d, f["name"], f["index"]) for a in [".5", "2"]
                      for r in [".025", ".1", ".5", ".65", ".84", ".95", ".995"]
                      for d in [0, 1] for f in registry["State_constructor"]["family_definitions"]]
    actual_bases = [(b["a"], b["radius"], b["direction"], b["family"], b["family_index"]) for b in registry["bases"]]
    if actual_bases != expected_bases or [b["id"] for b in registry["bases"]] != ["B%03d" % i for i in range(112)]:
        raise RuntimeError("base registry/order mismatch")
    variants = registry["zero_primal_gradient_variants"]["seed_ids"]
    if variants != ["alpha-gradient-only", "chi-gradient-only", "A0-balanced-plus-gradients", "all-used-mixed"]:
        raise RuntimeError("variant registry mismatch")
    counts = {"bases": len(actual_bases), "base_dual_rows": len(actual_bases) * len(expected_seed_names),
              "zero_gradient_variant_rows": len(actual_bases) * len(variants),
              "closed_positive_dual_rows": len(registry["closed_contexts"]) * len(registry["closed_seed_order_xi_alpha_xi_chi_xi_gradient"]),
              "legacy_negative_reuse_rows": len(registry["closed_contexts"]),
              "FD_representatives": 4 * 4, "FD_side_evaluations": 4 * 4 * 5 * 2}
    counts["physical_reference_dual_rows"] = counts["base_dual_rows"] + counts["zero_gradient_variant_rows"]
    counts["positive_dual_rows"] = counts["physical_reference_dual_rows"] + counts["closed_positive_dual_rows"]
    counts["records"] = counts["positive_dual_rows"] + counts["legacy_negative_reuse_rows"]
    counts["total_helper_evaluations"] = 2 * counts["positive_dual_rows"] + counts["FD_side_evaluations"]
    if counts != registry["counts"] or counts != recipe["fixed_counts"]:
        raise RuntimeError("fixed count mismatch")
    probe = (CAP / "probe.cpp").read_text()
    if any(token in probe for token in ["ConformalRHS(", "inner::Gauge(", "Consistent(u)"]):
        raise RuntimeError("unplanned compound/kernel/projection call")
    return {"passed_source_hash_extraction_AST_registry_readback": True,
            "captured_files": len(before["files"]), "protected_inputs": len(protected),
            "approved_plan_copies_byte_exact": True, "accepted_outer_headers_byte_exact": headers,
            "original_State_and_CastPoint_bodies_byte_exact": True,
            "source001_probe_and_oracle_byte_exact": True,
            "stdlib_AST_parses": 3, "fixed_counts": counts,
            "candidate_or_oracle_imported_or_executed": False,
            "MP_Fraction_targets_recomputed": False, "compiler_or_queries": False,
            "scientific_saved_payload_decoded": False}


if __name__ == "__main__":
    report = main()
    (HERE / "source-readback.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True))
