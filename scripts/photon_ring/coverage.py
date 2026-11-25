"""Check every leaf touching the closed launch disk against the finest native spacing."""

from math import hypot, isclose


def audit(bounds, meta):
    c, w = meta["center"], meta["w"]
    if c - w < 0 or max(b[1] for b in bounds) < c + w or max(b[3] for b in bounds) < w:
        raise ValueError("launch disk extends outside domain")
    touched = []
    for x0, x1, z0, z1 in bounds:
        if hypot(max(x0 - c, 0, c - x1), max(z0, 0, -z1)) <= w:
            dx, dz = (x1 - x0) / 36, (z1 - z0) / 36
            if not (isclose(dx, 0.0625) and isclose(dz, 0.0625)):
                raise ValueError(
                    f"launch disk intersects coarse block {(x0, x1, z0, z1)}"
                )
            touched.append([x0, x1, z0, z1])
    if not touched or len(bounds) != 76:
        raise ValueError("canonical mesh required")
    return dict(leaves=touched, cell_width=0.0625, full_width_cells=2 * w / 0.0625)
