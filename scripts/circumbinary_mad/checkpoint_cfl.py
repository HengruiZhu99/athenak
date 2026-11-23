"""Locate the native face light-cone timestep bound on a prescribed FP64 MHD checkpoint."""

import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from magnetic_smooth_state import checkpoint_state


def limits(path):
    state = checkpoint_state(path)
    params = state["parameters"]
    cfl = float(params["time/cfl_number"])
    with path.open("rb") as src:
        header = src.read(65536)
        base = header.index(b"<par_end>\n") + 10
        count, root_level = struct.unpack_from("<ii", header, base)
        lo, hi = np.array(struct.unpack_from("<6d", header, base + 8)).reshape(2, 3)
        root_n = np.array(struct.unpack_from("<3i", header, base + 84))
        ng, *block_n = struct.unpack_from("<4i", header, base + 156)
        if min(block_n) < 2 or ng < 1:
            raise ValueError("Expected a three-dimensional mesh with halo cells")
        src.seek(base + 252)
        locations = np.fromfile(src, dtype="<i4", count=4 * count).reshape(-1, 4)
        src.seek(base + 252 + 20 * count)
        stride = struct.unpack("<Q", src.read(8))[0]
        data = src.tell()
        shape = np.array(block_n[::-1]) + 2 * ng
        cells = int(np.prod(shape))
        # The 17 ADM variables are the trailing component of each block record.
        adm_offset = stride - 17 * cells * 8
        if adm_offset <= 0:
            raise ValueError("Unexpected MHD/ADM restart record")
        active = tuple(slice(ng, ng + n) for n in block_n[::-1])
        best = {}
        for idx, loc in enumerate(locations):
            level = int(loc[3]) - root_level
            dx = (hi - lo) / (root_n * 2**level)
            origin = lo + loc[:3] * block_n * dx
            src.seek(data + idx * stride + adm_offset)
            fields = np.fromfile(src, dtype="<f8", count=17 * cells).reshape((17, *shape))
            center = fields[(slice(None), *active)]
            for d in range(3):
                for side in (-1, 1):
                    shifted = list(active)
                    shifted[2 - d] = slice(ng + side, ng + side + block_n[d])
                    other = fields[(slice(None), *shifted)]
                    xx, xy, xz, yy, yz, zz = 0.5 * (center[:6] + other[:6])
                    cof = (yy * zz - yz**2, xx * zz - xz**2, xx * yy - xy**2)
                    det = xx * cof[0] + xy * (xz * yz - xy * zz) + xz * (xy * yz - xz * yy)
                    alpha = 0.5 * (center[13] + other[13])
                    shift = 0.5 * (center[14 + d] + other[14 + d])
                    normal = alpha * np.sqrt(cof[d] / det)
                    speed = np.abs(shift) + normal
                    at = np.unravel_index(speed.argmax(), speed.shape)
                    dt = float(cfl * dx[d] / speed[at])
                    if not np.isfinite(dt) or dt <= 0:
                        raise ValueError("Invalid checkpoint geometry")
                    if level not in best or dt < best[level]["dt"]:
                        xyz = origin + (np.array(at[::-1]) + 0.5) * dx
                        best[level] = {
                            "dt": dt,
                            "direction": d + 1,
                            "side": side,
                            "xyz": xyz.tolist(),
                            "radius": float(np.hypot(*xyz[:2])),
                            "shift": float(shift[at]),
                            "normal_light_speed": float(normal[at]),
                            "dx": float(dx[d]),
                        }
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    print(json.dumps(limits(p.parse_args().checkpoint), indent=2))
