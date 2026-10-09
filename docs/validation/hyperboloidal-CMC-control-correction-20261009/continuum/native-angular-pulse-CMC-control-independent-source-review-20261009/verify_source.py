"""Standard-library source/recipe readback; never import the candidate."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPTURE = HERE / "captured"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def main():
    before = json.loads((HERE / "inputs-before-review.json").read_text())
    for item in before["files"]:
        if sha(item["path"]) != item["sha256"] or sha(HERE / item["captured"]) != item["sha256"]:
            raise RuntimeError("source/copy drift: " + item["path"])
    for item in before["protected_inputs"]:
        if sha(item["path"]) != item["sha256"]:
            raise RuntimeError("protected input drift: " + item["path"])
    index = json.loads((CAPTURE / "source-index.json").read_text())
    recipe = json.loads((CAPTURE / "control-recipe.json").read_text())
    old_recipe = json.loads((CAPTURE / "source-history/original-full-recipe.json").read_text())
    settings = recipe["scientific_settings"]
    if any(value != old_recipe[key] for key, value in settings.items()):
        raise RuntimeError("original scientific settings changed")
    if {**recipe["protected_pins"], **recipe["mpmath_python_pins"]} != index["protected_inputs"]:
        raise RuntimeError("index/recipe protected closure mismatch")
    if sha(recipe["runtime_path"]) != recipe["runtime_sha256"]:
        raise RuntimeError("interpreter pin mismatch")
    old = (CAPTURE / "source-history/derivative_core-original.py").read_bytes()
    new = (CAPTURE / "derivative_core.py").read_bytes()
    oldline = b"            b=omega*q/self.a\n"
    newline = b'            b=omega*(sumjet(t*t for t in Y)**mp.mpf(".5"))/self.a\n'
    if old.count(oldline) != 1 or new.count(newline) != 1 or new.replace(newline, oldline) != old:
        raise RuntimeError("more than the one radius-jet line changed")
    dump = lambda text: ast.dump(ast.parse(text), include_attributes=False)
    if dump(new.replace(newline, oldline)) != dump(old):
        raise RuntimeError("reverse AST mismatch")
    trees = {name: ast.parse((CAPTURE / name).read_text()) for name in
             ["controls_only.py", "qualify_saved_checks.py", "derivative_core.py", "analytic_jets.py", "values_context.py"]}
    original_tree = ast.parse(old)
    nodes = lambda tree: {node.name: node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    oldnodes, newnodes = nodes(original_tree), nodes(trees["derivative_core.py"])
    changed = [name for name in oldnodes if ast.dump(oldnodes[name], include_attributes=False) != ast.dump(newnodes[name], include_attributes=False)]
    if changed != ["ControlGraph"]:
        raise RuntimeError("unexpected changed top-level definitions")
    oldmethods, newmethods = nodes(oldnodes["ControlGraph"]), nodes(newnodes["ControlGraph"])
    changed_methods = [name for name in oldmethods if ast.dump(oldmethods[name], include_attributes=False) != ast.dump(newmethods[name], include_attributes=False)]
    if changed_methods != ["source"]:
        raise RuntimeError("unexpected changed control methods")
    rows = len(settings["precisions"]) * len(settings["levels"]) * len(settings["controls"]) * len(settings["control_boosts"])
    rays = len(settings["precisions"]) * len(settings["controls"]) * len(settings["control_boosts"]) * sum(
        item["polar_order"] * item["azimuth_order"] for item in settings["levels"])
    final_rows = len(settings["precisions"]) * len(settings["controls"]) * len(settings["control_boosts"])
    checks = rows * 8 + final_rows * 4
    if (rows, rays, checks) != (100, 189440, 880):
        raise RuntimeError("fixed registry count mismatch")
    if (recipe["fixed_rows"], recipe["fixed_roots"], recipe["fixed_checks"]) != (rows, rays, checks):
        raise RuntimeError("recipe count mismatch")
    keys = ["root", "null", "first_graph", "second_graph", "factor_quotient", "K_jet", "D_source_jet", "Lorentz_matrix"]
    expected = set()
    for dps in settings["precisions"]:
        for level in settings["levels"]:
            for control in settings["controls"]:
                for boost in settings["control_boosts"]:
                    prefix = "control/%s/%s/%s/%s/" % (dps, level["name"], control["name"], boost)
                    expected.update(prefix + key for key in keys)
                    if level["name"] == settings["final_level"]:
                        expected.update("exact/%s/%s/%s/%s" % (dps, control["name"], boost, block)
                                        for block in ["u", "gradient", "hessian"])
                        expected.add("wave/control/%s/%s/%s/%s" % (dps, level["name"], control["name"], boost))
    if len(expected) != checks or 2360 - checks != recipe["saved_other_check_count"]:
        raise RuntimeError("saved/control partition count mismatch")
    # All candidate parsing above is AST only. No module execution or saved
    # scientific check/jet classification is performed by this reviewer.
    return {
        "passed_source_recipe_hash_readback": True,
        "captured_files": len(before["files"]), "protected_inputs": len(before["protected_inputs"]),
        "candidate_ASTs_parsed_not_executed": len(trees),
        "single_literal_change_and_reverse_AST_identical": True,
        "changed_top_level_definitions": changed, "changed_control_methods": changed_methods,
        "original_scientific_settings_equal": True,
        "index_recipe_protected_closure_equal": True, "runtime_bytes_pinned": True,
        "control_rows": rows, "analytic_control_rays": rays,
        "local_checks": rows * 8, "exact_block_checks": final_rows * 3,
        "wave_trace_checks": final_rows, "fixed_control_checks": checks,
        "expected_control_names": len(expected), "other_saved_checks_partition": 2360 - checks,
        "candidate_imported_or_executed": False, "scientific_targets_recomputed": False,
        "saved_scientific_payload_decoded": False,
        "original_full_gate_remains_failed": True,
    }


if __name__ == "__main__":
    result = main()
    (HERE / "source-readback.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True))
