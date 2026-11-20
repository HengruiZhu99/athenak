"""Run SpECTRE preprocessing and CCE on a converted AthenaK worldtube."""

import argparse
import subprocess
from pathlib import Path

import h5py
import yaml


def run(metric, release, output, config=None):
    """Use the release's matching example options unless a CCE input is supplied."""
    release = release.resolve()
    with h5py.File(metric) as f:
        radius = float(f.attrs["ExtractionRadius"])
        modes = (f["Lapse.dat"].shape[1] - 1) // 2
        lmax = int(modes**0.5) - 1
    output.mkdir(parents=True, exist_ok=False)
    pre = release / "PreprocessCceWorldtube"
    params = yaml.safe_load((pre / "PreprocessCceWorldtube.yaml").read_text())
    params.update(
        InputH5File=str(metric.resolve()),
        OutputH5File="bondi.h5",
        InputDataFormat="MetricModal",
        ExtractionRadius=radius,
    )
    (output / "preprocess.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
    subprocess.run(
        [pre / "PreprocessCceWorldtube", "--input-file", "preprocess.yaml"],
        cwd=output,
        check=True,
    )
    docs = list(
        yaml.safe_load_all((config or release / "CharacteristicExtract.yaml").read_text())
    )
    cce = docs[-1]["Cce"]
    cce.update(BoundaryDataFilename="bondi.h5", ExtractionRadius=radius)
    if config is None:
        cce["ObservationLMax"] = lmax
        cce["LMax"] = max(cce["LMax"], lmax + 6)
        cce["Filtering"]["FilterLMax"] = cce["LMax"] - 2
        cce["H5Interpolator"]["BarycentricRationalSpanInterpolator"] = {
            "MinOrder": 4,
            "MaxOrder": 4,
        }
    (output / "cce.yaml").write_text(yaml.safe_dump_all(docs, sort_keys=False))
    # Some Charm++ failure paths print a normal shutdown banner.
    log = output / "evolution.log"
    with log.open("x") as stream:
        subprocess.run(
            [release / "CharacteristicExtract", "+p1", "--input-file", "cce.yaml"],
            cwd=output,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    reduction = output / (docs[-1]["Observers"]["ReductionFileName"] + ".h5")
    if "PostFailureCleanup" in log.read_text() or not reduction.is_file():
        raise RuntimeError(f"CCE failed to finish. See {log}")
    return reduction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("metric", "release", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--config", type=Path, help="custom CharacteristicExtract YAML")
    a = p.parse_args()
    run(a.metric, a.release, a.output, a.config)


if __name__ == "__main__":
    main()
