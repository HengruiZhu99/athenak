"""Static source/diff review; never import or execute the growth analyzer."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
CONTINUUM = HERE.parents[1]
OLD = CONTINUUM / "finite-rb-limited-matrix-growth-20261009/analyze_growth.py"
NEW = CONTINUUM / "finite-rb-limited-matrix-growth-pade13-20261009/analyze_growth.py"
OUT = HERE / "analyzer-source-review-001"
OUT.mkdir(exist_ok=False)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def functions(path):
    return {node.name: node for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef)}


old, new = functions(OLD), functions(NEW)
assert set(new) - set(old) == {"expm"}
assert set(old) - set(new) == set()
unchanged = []
for name in old:
    if name != "analyze":
        assert ast.dump(old[name]) == ast.dump(new[name]), name
        unchanged.append(name)
old_try = [node for node in old["analyze"].body if isinstance(node, ast.Try)]
new_try = [node for node in new["analyze"].body if isinstance(node, ast.Try)]
assert len(old_try) == len(new_try) == 1
assert ast.dump(old_try[0]) == ast.dump(new_try[0])
assert sha(OLD) == "7634bf135c20df9588f2e6cc33dd434d6e63925e6716586445443033dc7224e3"
assert sha(NEW) == "cf51853026bd8647ff497559f2b741db17a8cb4c7653fa164f9096523099f2a0"
assert sha(NEW.parent / "EXPONENTIAL-ADDENDUM.md") == "4505c4ce343697c9616a7cfd99935ad6682eeb06fedef39d360182964d028cc8"
assert sha(NEW.parent / "pade13_einsum.py") == sha(HERE / "pade13_einsum.py")
assert sha(HERE / "pade13_einsum.py") == "0654c3ba8884dc467aa1c4665fc4758a3abddf58189e2702f7e8cfc90d2e79b7"
pins = {"review_analyzer.py": sha(Path(__file__)), "old_analyzer": sha(OLD),
        "new_analyzer": sha(NEW), "helper": sha(HERE / "pade13_einsum.py"),
        "exponential_addendum": sha(NEW.parent / "EXPONENTIAL-ADDENDUM.md")}
for name in ("PLAN.md", "FINAL-ADDENDUM.md", "LIMITED-MATRIX-SCOPE.md"):
    assert (OLD.parent / name).read_bytes() == (NEW.parent / name).read_bytes()
    pins[name] = sha(NEW.parent / name)
attempts = {}
for folder in ("synthetic-001", "large-norm-001", "failure-runtime-001"):
    p = HERE / folder / "results/receipt.json"
    data = json.loads(p.read_text())
    assert data["status"] == "PASS"
    attempts[folder] = {"path": str(p), "sha256": sha(p)}
    command = json.loads((HERE / folder / "command-receipt.json").read_text())
    assert command["returncode"] == 0
    assert (HERE / folder / "stderr.txt").read_bytes() == b""
diff = subprocess.run(["diff", "-u", str(OLD), str(NEW)], capture_output=True, text=True)
assert diff.returncode == 1
(OUT / "analyzer.diff").write_text(diff.stdout)
sources = OUT / "reviewed-sources"
sources.mkdir()
for target, path in (("old-analyzer.py", OLD), ("new-analyzer.py", NEW),
                     ("pade13_einsum.py", HERE / "pade13_einsum.py"),
                     ("EXPONENTIAL-ADDENDUM.md", NEW.parent / "EXPONENTIAL-ADDENDUM.md")):
    shutil.copyfile(path, sources / target)
result = {
    "status": "PASS_source_and_scope_review", "source_sha256": pins,
    "synthetic_gate_pins": attempts, "unchanged_functions_ast": unchanged,
    "complete_numeric_try_except_body_ast_identical": True,
    "source_operation": "replace SciPy expm binding with copied reviewed fixed-Pade13 wrapper only",
    "admission_changes": "mandatory helper/addendum review pins; exponential diagnostics and historical failure flag",
    "wrapper_review": [
        "Each CLI exponential call records scaling, rational solve residual, and status.",
        "The existing algebra threshold 2e-9 applies only to the rational solve residual; it is not an exponential error bound.",
        "Helper enforces all NumPy floating errors, including underflow, and performs no fallback or suppression.",
        "Original numerical E transform, eigensolve, physical seeds, exp times, half-time consistency, RK3 and guards are unchanged.",
        "Original failed SciPy attempt remains failed, with no propagation acceptance.",
        "EXPM_DIAGNOSTICS is a per-process list; normal reviewed admission is one CLI analyze call per process."
    ],
    "corrections": [], "analyzer_imported_or_executed": False,
    "scientific_operator_loaded": False,
    "scope": "Supports parent consideration of a separately admitted J0/N8 finite-matrix replay only; no PDE/native/nonlinear/scri/BH acceptance."
}
(OUT / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"status": result["status"], "receipt_sha256": sha(OUT / "receipt.json")}))
