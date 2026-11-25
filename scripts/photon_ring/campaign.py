"""Paths and launch settings shared by the photon-ring campaign stages."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PHOTON_CAMPAIGN", ".")).resolve()
SOURCE = Path(os.environ.get("PHOTON_SOURCE", CODE.parents[1])).resolve()
DIAGNOSTICS = SOURCE / "tst/unit/boltzmann/equator"
sys.path[1:1] = map(str, (DIAGNOSTICS, DIAGNOSTICS.parent, DIAGNOSTICS / "smr"))


def load(name):
    return json.loads((ROOT / name).read_text())


def case_source(key):
    archived = load("scan.json").get("archived", {}).get(key)
    return ROOT / archived if archived else ROOT / "cases" / key


@lru_cache(maxsize=4)
def executable_digest(path, mtime_ns, size):
    # Include file metadata in the cache key if a binary is replaced between segments.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def solver_command(parameters):
    profile = load("cluster.json")
    launcher = profile["launcher"]
    if not isinstance(launcher, list) or not all(
        isinstance(arg, str) for arg in launcher
    ):
        raise ValueError("launcher must be an array of command arguments")
    prefix = [s.replace("{source}", str(SOURCE)) for s in launcher]
    exe = (ROOT / profile["executable"]).resolve(strict=True)
    if not os.access(exe, os.X_OK):
        raise ValueError(f"AthenaK executable is not executable: {exe}")
    expected = profile.get("executable_sha256")
    stat = exe.stat()
    if expected and executable_digest(exe, stat.st_mtime_ns, stat.st_size) != expected:
        raise ValueError(f"AthenaK executable differs from the campaign digest: {exe}")
    return [*prefix, str(exe), *parameters]


def angular_grid(meta):
    if meta["allow_counterrotation"]:
        return "both"
    return "retrograde" if meta["J_z"] < 0 else "prograde"
