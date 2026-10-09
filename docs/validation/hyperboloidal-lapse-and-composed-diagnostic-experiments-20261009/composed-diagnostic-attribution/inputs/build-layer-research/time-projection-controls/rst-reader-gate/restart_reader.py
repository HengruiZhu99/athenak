"""Read-only native RST reader for one uniform vacuum hyperboloidal Z4c block.

This intentionally rejects other physics, trackers, refinement and other ABIs.
Arrays are little-endian binary64, read-only, including ghosts, LayoutRight
(meshblock,variable,k,j,i). This is a parser, not a restart/evolution action.
"""
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

import numpy as np

VARIABLES = (
    "z4c_chi", "z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy", "z4c_gyz",
    "z4c_gzz", "z4c_Khat", "z4c_Axx", "z4c_Axy", "z4c_Axz", "z4c_Ayy",
    "z4c_Ayz", "z4c_Azz", "z4c_Gamx", "z4c_Gamy", "z4c_Gamz", "z4c_Theta",
    "z4c_alpha", "z4c_betax", "z4c_betay", "z4c_betaz", "z4c_Bx", "z4c_By", "z4c_Bz")
REGION = ("x1min", "x2min", "x3min", "x1max", "x2max", "x3max", "dx1", "dx2", "dx3")
INDICES = ("ng", "nx1", "nx2", "nx3", "is", "ie", "js", "je", "ks", "ke",
           "cnx1", "cnx2", "cnx3", "cis", "cie", "cjs", "cje", "cks", "cke")
LOGICAL = ("lx1", "lx2", "lx3", "level")


def parameters(text):
    """The final assignment wins, matching ParameterInput semantics."""
    values, section = {}, ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("<") and line.endswith(">"):
            section = line[1:-1]
        elif "=" in line:
            key, value = line.split("=", 1)
            values[section+"/"+key.strip()] = value.strip()
    return values


def abi_check(abi):
    expected = {"Real": 8, "int": 4, "float": 4, "IOWrapperSizeT": 8,
                "RegionSize": 72, "RegionIndcs": 76, "LogicalLocation": 16,
                "nz4c": 25, "little_endian": 1, "LayoutRight": 1}
    if sys.byteorder != "little" or any(abi.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported native RST ABI")
    for name, fields, width in [("RegionSize", REGION, 8), ("RegionIndcs", INDICES, 4),
                                ("LogicalLocation", LOGICAL, 4)]:
        if abi[name+"_offsets"] != {field: i*width for i, field in enumerate(fields)}:
            raise ValueError("unsupported struct member offsets")
    roots = [parent for parent in Path(__file__).resolve().parents
             if (parent/"src/outputs/restart.cpp").is_file()]
    if not roots or not abi.get("source_sha256"):
        raise ValueError("cannot verify production source layout")
    root = roots[0]
    for relative, digest in abi["source_sha256"].items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest() != digest:
            raise ValueError("production source/ABI hash changed: "+relative)
    source = (root/"src/z4c/z4c.cpp").read_text()
    section = source.split("Z4c::Z4c_names[Z4c::nz4c] = {", 1)[1].split("};", 1)[0]
    if tuple(re.findall(r'"([^"]+)"', section)) != VARIABLES:
        raise ValueError("production Z4c variable order changed")


def read_restart(filename, abi=None):
    """Return exact header metadata and a read-only memmap, without rounding.

    result['data']: (1,25,nz+2ng,ny+2ng,nx+2ng), result['mb_data'][name]:
    (1,nz+2ng,ny+2ng,nx+2ng). result['active_data'] excludes box ghosts;
    the spherical PDE mask must still be applied for hyperboloidal analysis.
    header dt is pm->dt when output is written, not invariably last step dt.
    """
    path = Path(filename).resolve()
    if abi is None:
        abi = json.loads(Path(__file__).with_name("abi.json").read_text())
    abi_check(abi)
    with path.open("rb") as stream:
        prefix = stream.read(40960)
        marker = prefix.find(b"<par_end>\n")
        if marker < 0:
            raise ValueError("ParameterInput terminator absent from first40KiB")
        text_end = marker+10
        text = prefix[:text_end].decode("ascii")
        pin = parameters(text)
        if pin.get("z4c/hyperboloidal") not in ("true", "1"):
            raise ValueError("requires the vacuum hyperboloidal Z4c case")
        if any(key.split("/")[0] in ("hydro", "mhd", "radiation", "turbulence", "gravity", "particles")
               or re.fullmatch(r"z4c/co_\d+_type", key) for key in pin):
            raise ValueError("other physics/tracker RST layout is unsupported")
        if pin.get("mesh_refinement/refinement", "none") != "none":
            raise ValueError("refinement is unsupported")
        offsets = {"parameter_header": 0, "parameter_header_end": text_end}
        cursor = text_end

        def unpack(name, fmt):
            nonlocal cursor
            offsets[name] = cursor
            size = struct.calcsize("<"+fmt)
            stream.seek(cursor)
            data = stream.read(size)
            if len(data) != size:
                raise ValueError("truncated RST header")
            cursor += size
            return struct.unpack("<"+fmt, data)

        nmb, = unpack("nmb_total", "i")
        root_level, = unpack("root_level", "i")
        if nmb != 1 or root_level != 0:
            raise ValueError("requires one uniform root MeshBlock")
        region = dict(zip(REGION, unpack("mesh_size", "9d")))
        mesh = dict(zip(INDICES, unpack("mesh_indcs", "19i")))
        block = dict(zip(INDICES, unpack("mb_indcs", "19i")))
        time, = unpack("time", "d")
        dt, = unpack("dt", "d")
        cycle, = unpack("ncycle", "i")
        logical = dict(zip(LOGICAL, unpack("logical_location", "4i")))
        cost, = unpack("cost", "f")
        last_output_time, = unpack("z4c_last_output_time", "d")
        data_size, = unpack("data_size", "Q")
        offsets["data"] = cursor
    if any(mesh[key] != block[key] for key in INDICES[:10]) or \
            any(mesh[key] != 0 for key in INDICES[10:]) or \
            logical != dict.fromkeys(LOGICAL, 0):
        raise ValueError("uniform single block dimensions/location mismatch")
    ng = block["ng"]
    if ng <= 0 or any(block[f"nx{i}"] <= 1 for i in range(1, 4)):
        raise ValueError("requires a 3D ghosted block")
    for axis, left, right in [(1, "is", "ie"), (2, "js", "je"), (3, "ks", "ke")]:
        n = block[f"nx{axis}"]
        if block[left] != ng or block[right] != ng+n-1:
            raise ValueError("invalid active storage indices")
        coarse_left, coarse_right = ("c"+left, "c"+right)
        if n % 2 or block[f"cnx{axis}"] != n//2 or block[coarse_left] != ng or \
                block[coarse_right] != ng+n//2-1:
            raise ValueError("invalid uniform coarse storage indices")
        for section in ("mesh", "meshblock"):
            if int(pin[f"{section}/nx{axis}"]) != n:
                raise ValueError("input/header dimensions mismatch")
        if int(pin["mesh/nghost"]) != ng:
            raise ValueError("input/header ghost count mismatch")
        if region[f"x{axis}min"] != float(pin[f"mesh/x{axis}min"]) or \
                region[f"x{axis}max"] != float(pin[f"mesh/x{axis}max"]):
            raise ValueError("input/header region mismatch")
        expected_dx = (region[f"x{axis}max"]-region[f"x{axis}min"])/n
        if region[f"dx{axis}"] != expected_dx:
            raise ValueError("region/header spacing mismatch")
    shape = (1, 25, block["nx3"]+2*ng, block["nx2"]+2*ng, block["nx1"]+2*ng)
    expected_size = int(np.prod(shape))*8
    if data_size != expected_size or path.stat().st_size != cursor+expected_size:
        raise ValueError("RST payload size/EOF mismatch (incomplete or other physics)")
    if not np.isfinite([time, dt, cost, last_output_time]).all() or time < 0 or dt <= 0 or cycle < 0:
        raise ValueError("invalid exact RST header scalars")
    data = np.memmap(path, dtype="<f8", mode="r", offset=cursor, shape=shape, order="C")
    active = (slice(None), slice(None), slice(block["ks"], block["ke"]+1),
              slice(block["js"], block["je"]+1), slice(block["is"], block["ie"]+1))
    return {"path": str(path), "header": text, "parameters": pin,
            "time": time, "dt": dt, "cycle": cycle, "offsets": offsets,
            "n_mbs": nmb, "root_level": root_level, "mesh_size": region,
            "mesh_indcs": mesh, "mb_indcs": block, "logical_location": logical,
            "cost": cost, "z4c_last_output_time": last_output_time,
            "data_size": data_size, "shape": shape, "var_names": VARIABLES,
            "data": data, "active_data": data[active],
            "mb_data": {name: data[:, i] for i, name in enumerate(VARIABLES)}}


def align_with_bin(restart, binary):
    """Read-only RST array view matching this one-block BIN's storage range."""
    if binary["n_mbs"] != 1 or not np.array_equal(binary["mb_logical"], [[0, 0, 0, 0]]):
        raise ValueError("BIN requires the same one root block")
    ng = restart["mb_indcs"]["ng"]
    bounds = np.asarray(binary["mb_index"])[0]+ng
    slices = []
    for axis in range(3):
        low, high = map(int, bounds[2*axis:2*axis+2])
        n = restart["mb_indcs"][f"nx{axis+1}"]
        if low < 0 or high >= n+2*ng or high < low or binary[f"nx{axis+1}_mb"] != n:
            raise ValueError("BIN/RST dimensions or range mismatch")
        region = restart["mesh_size"]
        if not np.array_equal(binary["mb_geometry"][0][2*axis:2*axis+2],
                              [region[f"x{axis+1}min"], region[f"x{axis+1}max"]]):
            raise ValueError("BIN/RST geometry mismatch")
        slices.append(slice(low, high+1))
    return restart["data"][(slice(None), slice(None), *slices[::-1])]


def read_rst(path, abi=None):
    """Root audit API: binary64 u has shape(25,nz,ny,nx), including ghosts.

    u is a read-only ndarray view. All exact binary metadata and the parsed
    section/key parameter map are retained. See read_restart for scope checks.
    """
    result = read_restart(path, abi)
    result["u"] = result["data"][0]
    return result
