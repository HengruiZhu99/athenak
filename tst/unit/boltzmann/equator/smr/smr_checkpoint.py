"""Validate the layout and epoch of a published SMR photon-ring restart."""

import struct


def validate_restart(path, expected_time=None, expected_cycle=None):
    with path.open("rb") as file:
        header = file.read(256 * 1024)
        begin = header.index(b"<par_end>\n") + len(b"<par_end>\n")
        block = header[:begin].split(b"<boltzmann>\n", 1)[1].split(b"<", 1)[0]
        params = {}
        for line in block.splitlines():
            name, sep, value = line.partition(b"=")
            if sep:
                params[name.strip()] = value.split(b"#")[0].strip()
        signs = {b"prograde": 1, b"retrograde": 1, b"both": 2}[params[b"sign"]]
        bins = int(params[b"nmu_half"]) * int(params[b"npsi"]) * signs
        stride = 8 * (bins * 44**2 + 25 * 45**2)
        offset = path.stat().st_size - 76 * stride
        if not begin < offset < 4 * 1024 * 1024:
            raise RuntimeError(f"incomplete checkpoint {path}")
        (nblocks,) = struct.unpack_from("=i", header, begin)
        # Two leading integers, RegionSize (9 doubles), two RegionIndcs (19 integers each).
        t, dt, cycle = struct.unpack_from(
            "=ddi", header, begin + 8 + 9 * 8 + 2 * 19 * 4
        )
        if (
            nblocks != 76
            or (expected_cycle is not None and cycle != expected_cycle)
            or (expected_time is not None and abs(t - expected_time) > 1.0e-8)
        ):
            raise ValueError(f"checkpoint epoch mismatch {path}: t={t}, cycle={cycle}")
        file.seek(offset - 8)
        if struct.unpack("=Q", file.read(8))[0] != stride:
            raise RuntimeError(f"bad checkpoint field stride {path}")
        return t, cycle
