"""Saved certificate replay: exact coverage plus an independent Gaussian chart."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from admission import check, digest
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    _, recipe, authorization, index_sha, _ = check(args.recipe, args.authorization, args.output, "replay")
    from fractions import Fraction as Q
    from interval import Context, decode, directed, pow2
    from producer_bounds import coefficients
    from replay_bounds import lower_bound

    out = Path(args.output)
    if (out / "report.json").exists():
        raise RuntimeError("replay output already exists")
    pin = authorization["certificate_payload"]
    if pin["bytes"] > recipe["max_certificate_bytes"]:
        raise RuntimeError("certificate byte limit")
    ctx, independent_ctx = Context(recipe["bits"]), Context(recipe["replay_bits"])
    K = recipe["series_order"]
    roots = []
    for sigma_text in recipe["sigma"]:
        sigma = Q(sigma_text)
        roots.extend([(sigma, (Q(1, 50), 2 * sigma, Q(0), Q(4) + 8 * sigma)),
                      (sigma, (2 * sigma, Q(4), Q(0), Q(4) + 8 * sigma))])
    stack = [(i, "", b, 0) for i, (_, b) in reversed(list(enumerate(roots)))]
    started = time.monotonic()
    nodes, leaves, minimum, producer_minimum = 0, 0, None, None
    counts = {"regular": 0, "separated": 0, "upper_tail": 0, "lower_tail": 0}
    with Path(pin["path"]).open() as handle:
        header = json.loads(next(handle))
        expected_roots = [[str(s), [str(x) for x in b]] for s, b in roots]
        if not (header.get("kind") == "header" and header.get("source_index_sha256") == index_sha
                and header.get("roots") == expected_roots and header.get("bits") == ctx.bits
                and header.get("series_order") == K):
            raise RuntimeError("wrong certificate header")
        while stack:
            if time.monotonic() - started > recipe["replay_wall_seconds"]:
                raise RuntimeError("UNRESOLVED: replay time limit")
            raw = next(handle)
            if len(raw) > 8192:
                raise RuntimeError("oversized node")
            row = json.loads(raw)
            root_id, path, box, depth = stack.pop()
            if row.get("root") != root_id or row.get("path") != path:
                raise RuntimeError("tree coverage/order mismatch")
            nodes += 1
            if nodes > 2 * recipe["max_leaves"] + len(roots):
                raise RuntimeError("node limit")
            sigma, (rlo, rhi, tlo, thi) = roots[root_id][0], box
            if row.get("kind") == "split":
                axis = row.get("axis")
                if type(axis) is not int or axis not in (0, 1) or depth >= recipe["max_depth"]:
                    raise RuntimeError("invalid split")
                if axis != (0 if rhi - rlo >= thi - tlo else 1):
                    raise RuntimeError("split rule changed")
                if axis == 0:
                    mid = (rlo + rhi) / 2
                    left, right = (rlo, mid, tlo, thi), (mid, rhi, tlo, thi)
                else:
                    mid = (tlo + thi) / 2
                    left, right = (rlo, rhi, tlo, mid), (rlo, rhi, mid, thi)
                stack.extend([(root_id, path + "1", right, depth + 1), (root_id, path + "0", left, depth + 1)])
                continue
            if row.get("kind") != "leaf":
                raise RuntimeError("missing leaf")
            method = row.get("method")
            if method == "upper_tail":
                if not tlo - rhi >= 8 * sigma:
                    raise RuntimeError("invalid upper-tail coverage")
                own = (Q(1, 10) - Q(3, 4) * pow2(-19)) * Q(7, 157)
                expected = directed(own, ctx.bits, False)
            elif method == "lower_tail":
                if not rlo - thi >= 8 * sigma:
                    raise RuntimeError("invalid lower-tail coverage")
                own = (Q(1, 10) - 110 * pow2(-32)) * Q(7, 157)
                expected = directed(own, ctx.bits, False)
            else:
                # Producer binding check is separate from the independent
                # ordinary-dimensional/Horner/common-denominator enclosure.
                original = coefficients(ctx, box, sigma, K)
                if method != original["method"]:
                    raise RuntimeError("wrong producer method")
                witness = row.get("coefficient_witness", {})
                for name in ("mlo", "qlo", "Lhi"):
                    if decode(witness[name]) != original[name]:
                        raise RuntimeError("producer coefficient witness mismatch")
                expected = directed(original["lower"], ctx.bits, False)
                own = lower_bound(independent_ctx, box, sigma, K)
            if decode(row["lower"]) != expected or expected <= 0 or own <= 0:
                raise RuntimeError("UNRESOLVED/FAIL: positive independent leaf not established at " + path)
            leaves += 1
            if leaves > recipe["max_leaves"] or method not in counts:
                raise RuntimeError("leaf registry/limit")
            counts[method] += 1
            minimum = own if minimum is None else min(minimum, own)
            producer_minimum = expected if producer_minimum is None else min(producer_minimum, expected)
            if nodes % 128 == 0:
                (out / "progress.json").write_text(json.dumps({"nodes": nodes, "leaves": leaves,
                    "pending": len(stack), "elapsed_seconds": time.monotonic() - started}, allow_nan=False) + "\n")
        footer = json.loads(next(handle))
        if not (footer.get("kind") == "footer" and footer.get("nodes") == nodes
                and footer.get("leaves") == leaves and footer.get("counts") == counts
                and decode(footer.get("minimum_lower")) == producer_minimum):
            raise RuntimeError("footer/coverage mismatch")
        if handle.read().strip():
            raise RuntimeError("unconsumed certificate records")
    if digest(pin["path"]) != pin["sha256"]:
        raise RuntimeError("certificate changed during replay")
    from interval import encode
    report = {"passed": True, "stage": "replay", "source_index_sha256": index_sha,
              "nodes": nodes, "leaves": leaves, "counts": counts,
              "independent_minimum_lower": encode(directed(minimum, independent_ctx.bits, False)),
              "complete_tree_coverage": True, "independent_function_enclosures_positive": True,
              "certificate_sha256": pin["sha256"], "regional_pencils_still_required": True,
              "scope": "exact compact physical-event endpoint box; no PDE/native/jet/BH acceptance"}
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
