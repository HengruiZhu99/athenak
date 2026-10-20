"""Check fixed-ball constraint histories against independent 3D cell sums."""
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess

import numpy as np
import pytest

from .overhaul_utils import ROOT

pytest.importorskip("h5py")
spec = importlib.util.spec_from_file_location("history_bin", ROOT / "vis/python/bin_convert.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


@pytest.mark.parametrize("radius,start,buffer,inner", [
    (1.5, 0.0, 0.0, 0.0),
    (1.5, 10.0, 1.e6, 2.0),
    (10.0, 10.0, 1.e6, 2.0),
])
def test_fixed_ball_history(tmp_path, radius, start, buffer, inner):
    executable = str(Path(os.environ.get("ATHENA_OVERHAUL_EXE", "./athena")).resolve())
    overrides = ["job/basename=history", "time/nlim=0", f"time/start_time={start}",
                 "time/tlim=11", "mesh_refinement/refinement=none",
                 "z4c/horizon_finder=none", f"z4c/history_interior_radius={radius}",
                 f"z4c/history_inner_radius={inner}", f"z4c/history_boundary_buffer={buffer}",
                 "z4c/excise_chi=0.0625", "output1/data_format=%24.16e"]
    for axis in range(1, 4):
        overrides += [f"mesh/nx{axis}=16", f"meshblock/nx{axis}=8",
                      f"mesh/x{axis}min=-2", f"mesh/x{axis}max=2"]
    for number, variable in enumerate(("adm", "z4c", "con"), 2):
        overrides += [f"output{number}/file_type=bin", f"output{number}/variable={variable}",
                      f"output{number}/dt=1"]
    # Input overrides may introduce parameters absent from the supplied test deck.
    text = (ROOT / "tst/inputs/z4c_boosted.athinput").read_text()
    for override in overrides:
        parameter, value = override.split("=", 1)
        block, key = parameter.split("/", 1)
        text += f"\n<{block}>\n{key}={value}\n"
    source = tmp_path / "history.athinput"
    source.write_text(text)
    command = [executable, "-i", str(source)]
    result = subprocess.run(shlex.split(os.environ.get("ATHENA_TEST_LAUNCHER", "")) + command,
                            cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "history_inner_radius is ignored" in result.stderr
    fields = {}
    geometry = None
    for variable in ("adm", "z4c", "con"):
        path = sorted((tmp_path / "bin").glob(f"*.{variable}.*.bin"))[0]
        data = reader.read_binary(str(path))
        fields.update({key: np.asarray(value, dtype=float)
                       for key, value in data["mb_data"].items()})
        geometry = data["mb_geometry"]
    shape = fields["z4c_chi"].shape
    radius2 = np.empty(shape)
    coordinate_volume = np.empty(shape)
    for m, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(geometry):
        dx, dy, dz = ((xmax-xmin)/shape[3], (ymax-ymin)/shape[2],
                      (zmax-zmin)/shape[1])
        k, j, i = np.indices(shape[1:])
        radius2[m] = ((xmin+(i+0.5)*dx)**2 + (ymin+(j+0.5)*dy)**2
                      + (zmin+(k+0.5)*dz)**2)
        coordinate_volume[m] = dx*dy*dz
    xx, xy, xz, yy, yz, zz = [fields["adm_g"+component]
                             for component in ("xx", "xy", "xz", "yy", "yz", "zz")]
    determinant = xx*yy*zz + 2*xy*xz*yz - xx*yz**2 - yy*xz**2 - zz*xy**2
    volume = coordinate_volume * np.sqrt(np.abs(determinant))
    chi_pass = fields["z4c_chi"] >= 0.0625
    inside = radius2 < radius**2
    mask = inside & chi_pass
    assert mask.any() and (inside & ~chi_pass).any()
    history = np.atleast_2d(np.loadtxt(next(tmp_path.glob("*.z4c.user.hst"))))[0]
    # Binary outputs are float32; histories accumulate in Real (normally float64).
    for column, value in ((11, fields["con_H"]**2), (12, fields["con_M"]),
                          (13, fields["con_C"]), (14, np.ones(shape))):
        np.testing.assert_allclose(history[column], (volume*value)[mask].sum(), rtol=2.e-6)
    np.testing.assert_allclose(history[17], coordinate_volume[mask].sum(), rtol=1.e-13)
    np.testing.assert_allclose(history[18], volume[inside & ~chi_pass].sum(), rtol=2.e-6)
    if radius == 10:
        # No inner-radius hole and no time/boundary erosion, even with an unsafe envelope.
        np.testing.assert_allclose(history[[11, 12, 13, 14]], history[[3, 4, 2, 10]],
                                   rtol=1.e-13)
    if buffer > 0:
        assert history[15] > 0  # Boundary risk is recorded without excluding any cells.
