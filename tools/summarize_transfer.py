"""Summarize paired HCCL measurements without mixing warmups or profiler runs."""

import argparse
import json
from pathlib import Path

import numpy as np


def describe(values):
    lower, median, upper, p95 = np.percentile(values, [25, 50, 75, 95])
    return dict(n=len(values), median=float(median), iqr=float(upper-lower), p95=float(p95))


def summarize(directory):
    ranks = [json.loads(path.read_text()) for path in sorted(directory.glob("rank*.json"))]
    if not ranks or ranks[0]["task"] != "memory":
        return
    receiver = ranks[1]
    records = [record for record in receiver["records"] if record["sample"] >= 0]
    modes = sorted({record["mode"] for record in records})
    summary = {"physical_devices": [rank["physical_device"] for rank in ranks], "modes": {}}
    rng = np.random.default_rng(1701)
    baseline = {record["sample"]: record["seconds"] for record in records if record["mode"] == "A"}
    for mode in modes:
        rows = [record for record in records if record["mode"] == mode]
        result = dict(
            complete_transfer_and_copy_loader_seconds=describe([row["seconds"] for row in rows]),
            receiver_temporary_allocated_bytes=describe([row["transport_allocated_delta"] for row in rows]),
            max_collective_bytes=max(max(row["collective_bytes"], default=0) for row in rows),
            largest_parameter_bytes=rows[0]["largest_parameter_bytes"],
            bucket_bytes=rows[0]["bucket_bytes"], inventory_bytes=rows[0]["inventory_bytes"],
            all_bytes_correct=all(row["all_bytes_correct"] for row in rows),
        )
        pairs = [(baseline[row["sample"]], row["seconds"]) for row in rows if row["sample"] in baseline]
        if pairs and mode != "A":
            paired = np.array(pairs)
            indices = rng.integers(0, len(pairs), size=(10000, len(pairs)))
            bootstrap = np.median(paired[indices, 1], axis=1) / np.median(paired[indices, 0], axis=1) - 1
            result["paired_median_latency_change_ci95"] = np.percentile(bootstrap, [2.5, 97.5]).tolist()
            result["median_latency_change"] = float(np.median(paired[:, 1]) / np.median(paired[:, 0]) - 1)
        summary["modes"][mode] = result
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"directory": str(directory), **summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    for directory in parser.parse_args().directories:
        summarize(directory)
