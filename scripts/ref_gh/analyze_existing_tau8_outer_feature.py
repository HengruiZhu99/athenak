#!/usr/bin/env python3
"""Extract the old-domain ADM reversal onset from retained gate evidence."""

import argparse
import json
from pathlib import Path


METRICS = ("r2to4_H_L2", "r4to8_H_L2", "r2to4_M_L2", "r4to8_M_L2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gate_json", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.gate_json.read_text())
    records = []
    onsets = {metric: None for metric in METRICS}
    for record in source["records"]:
        item = {"time": record["target"], "metrics": {}}
        for metric in METRICS:
            values = {label: record["resolutions"][label][metric]
                      for label in ("coarse", "medium", "fine")}
            reversed_resolution = not (
                values["coarse"] >= values["medium"] >= values["fine"])
            item["metrics"][metric] = {
                "values": values, "resolution_reversed": reversed_resolution
            }
            if reversed_resolution and onsets[metric] is None:
                onsets[metric] = {"time": record["target"], "values": values}
        records.append(item)
    payload = {
        "schema": "ref-gh-existing-tau8-outer-feature-v1",
        "source": str(args.gate_json), "domain": "[-6M,6M]^3",
        "available_evidence": "fixed-shell norms at retained common times",
        "missing_evidence": [
            "common ADM maximum locations", "full retained histories in this checkout",
            "common ADM field outputs in this checkout"
        ],
        "classification": "D_unresolved_pending_enlarged_domain_comparison",
        "onsets": onsets, "records": records,
    }
    json_path = Path(str(args.output_prefix) + ".json")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tsv_path = Path(str(args.output_prefix) + ".tsv")
    with tsv_path.open("w") as stream:
        stream.write("time\tmetric\tcoarse\tmedium\tfine\tresolution_reversed\n")
        for record in records:
            for metric, info in record["metrics"].items():
                values = info["values"]
                stream.write("\t".join(str(value) for value in (
                    record["time"], metric, values["coarse"], values["medium"],
                    values["fine"], info["resolution_reversed"])) + "\n")
    print(json_path)
    print(tsv_path)


if __name__ == "__main__":
    main()
