"""Bounded exact compact-box certificate producer, gated and one-shot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from admission import check
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    _, recipe, _, index_sha, _ = check(args.recipe, args.authorization, args.output, "certificate")
    from fractions import Fraction as Q
    from interval import Context, directed, encode, pow2
    from producer_bounds import coefficients
    # BEGIN_OBSERVATION_ONLY
    from instrumentation import ObservedEndpointCache, ProgressObserver
    # END_OBSERVATION_ONLY

    out = Path(args.output)
    target = out / "certificate.jsonl"
    if target.exists() or (out / "report.json").exists():
        raise RuntimeError("certificate output already exists")
    ctx, K = Context(recipe["bits"]), recipe["series_order"]
    # BEGIN_OBSERVATION_ONLY
    cache = ObservedEndpointCache()
    ctx._exp_endpoint_cache = cache
    # END_OBSERVATION_ONLY
    roots = []
    for sigma_text in recipe["sigma"]:
        sigma = Q(sigma_text)
        for rlo, rhi in [(Q(1, 50), sigma), (sigma, Q(4))]:
            roots.append((sigma, (rlo, rhi, Q(0), Q(4) + 8 * sigma)))
    stack = [(i, "", box, 0) for i, (_, box) in reversed(list(enumerate(roots)))]
    started = time.monotonic()
    nodes, leaves, encoded_bytes = 0, 0, 0
    minimum = None
    transcript = hashlib.sha256()
    counts = {"regular": 0, "separated": 0, "upper_tail": 0, "lower_tail": 0}
    # BEGIN_OBSERVATION_ONLY
    observer = ProgressObserver(out, roots, started, cache)
    # END_OBSERVATION_ONLY

    def write(handle, row):
        nonlocal encoded_bytes
        raw = (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        encoded_bytes += len(raw)
        if encoded_bytes > recipe["max_certificate_bytes"]:
            raise RuntimeError("UNRESOLVED: declared certificate byte limit")
        handle.write(raw)
        transcript.update(raw)

    with target.open("xb") as handle:
        # BEGIN_OBSERVATION_ONLY
        observer.progress(nodes, leaves, stack, counts, minimum, "initial")
        # END_OBSERVATION_ONLY
        write(handle, {"kind": "header", "source_index_sha256": index_sha,
                       "roots": [[str(s), [str(x) for x in b]] for s, b in roots],
                       "bits": ctx.bits, "series_order": K,
                       "scope": "exact physical-event pure-CMC a2 compact endpoint box only"})
        while stack:
            if time.monotonic() - started > recipe["domain_wall_seconds"]:
                # BEGIN_OBSERVATION_ONLY
                observer.progress(nodes, leaves, stack, counts, minimum, "domain_time_cap")
                # END_OBSERVATION_ONLY
                raise RuntimeError("UNRESOLVED: declared domain time limit")
            root_id, path, box, depth = stack.pop()
            sigma = roots[root_id][0]
            rlo, rhi, tlo, thi = box
            nodes += 1
            if nodes > 2 * recipe["max_leaves"] + len(roots):
                raise RuntimeError("UNRESOLVED: node limit")
            # BEGIN_OBSERVATION_ONLY
            observer.active(root_id, path, box, depth)
            # END_OBSERVATION_ONLY
            result = None
            if tlo - rhi >= 8 * sigma:
                method = "upper_tail"
                low = (Q(1, 10) - Q(3, 4) * pow2(-19)) * Q(7, 157)
            elif rlo - thi >= 8 * sigma:
                method = "lower_tail"
                low = (Q(1, 10) - 110 * pow2(-32)) * Q(7, 157)
            else:
                result = coefficients(ctx, box, sigma, K)
                method, low = result["method"], result["lower"]
            low = directed(low, ctx.bits, False)
            # BEGIN_OBSERVATION_ONLY
            observer.lower(method, low)
            # END_OBSERVATION_ONLY
            if low > 0:
                leaves += 1
                if leaves > recipe["max_leaves"]:
                    raise RuntimeError("UNRESOLVED: leaf limit")
                counts[method] += 1
                # BEGIN_OBSERVATION_ONLY
                observer.leaf(root_id)
                # END_OBSERVATION_ONLY
                minimum = low if minimum is None else min(minimum, low)
                row = {"root": root_id, "path": path, "kind": "leaf", "method": method,
                       "lower": encode(low)}
                if result is not None:
                    row["coefficient_witness"] = {key: encode(result[key]) for key in ("mlo", "qlo", "Lhi")}
                write(handle, row)
            else:
                if depth >= recipe["max_depth"]:
                    raise RuntimeError("UNRESOLVED: depth limit")
                axis = 0 if rhi - rlo >= thi - tlo else 1
                if axis == 0:
                    mid = (rlo + rhi) / 2
                    left, right = (rlo, mid, tlo, thi), (mid, rhi, tlo, thi)
                else:
                    mid = (tlo + thi) / 2
                    left, right = (rlo, rhi, tlo, mid), (rlo, rhi, mid, thi)
                write(handle, {"root": root_id, "path": path, "kind": "split", "axis": axis})
                stack.append((root_id, path + "1", right, depth + 1))
                stack.append((root_id, path + "0", left, depth + 1))
            if nodes % 128 == 0:
                handle.flush()
                (out / "progress.json").write_text(json.dumps({"nodes": nodes, "leaves": leaves,
                    "pending": len(stack), "elapsed_seconds": time.monotonic() - started}, allow_nan=False) + "\n")
                # BEGIN_OBSERVATION_ONLY
                observer.progress(nodes, leaves, stack, counts, minimum, "checkpoint")
                # END_OBSERVATION_ONLY
        write(handle, {"kind": "footer", "nodes": nodes, "leaves": leaves,
                       "counts": counts, "minimum_lower": encode(minimum)})
    # BEGIN_OBSERVATION_ONLY
    observer.progress(nodes, leaves, stack, counts, minimum, "producer_finished")
    # END_OBSERVATION_ONLY
    report = {"passed": True, "stage": "certificate", "source_index_sha256": index_sha,
              "nodes": nodes, "leaves": leaves, "counts": counts,
              "minimum_compact_box_lower": encode(minimum), "certificate_bytes": encoded_bytes,
              "certificate_sha256": transcript.hexdigest(),
              "coverage_complete": True, "independent_replay_passed": False,
              "global_slicing_acceptance": False,
              "scope": "producer only; global interpretation requires independent replay and pinned regional pencils"}
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
