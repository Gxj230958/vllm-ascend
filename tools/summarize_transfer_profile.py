"""Measure HCCL-event and device-copy overlap in the independent profiler run."""

import csv
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1] / "experiments/bounded-transfer-20261102"


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


records = []
for rank in (0, 1):
    for mode in "ABC":
        paths = list((root / "profiler" / f"profiler-rank{rank}-{mode}").glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
        assert len(paths) == 1, (rank, mode, paths)
        with paths[0].open() as file:
            rows = list(csv.DictReader(file))
        comm, copies = [], []
        for row in rows:
            start = float(row["Start Time(us)"])
            interval = (start, start + float(row["Duration(us)"]))
            if row["Accelerator Core"] == "COMMUNICATION":
                comm.append(interval)
            elif "TensorMove" in row["Name"]:
                copies.append(interval)
        assert comm and copies, "Both HCCL and device copy events must be present"
        comm_intervals, copy_intervals = merge(comm), merge(copies)
        overlap = sum(max(0, min(b, d) - max(a, c)) for a, b in comm_intervals for c, d in copy_intervals)
        copy_time = sum(b - a for a, b in copy_intervals)
        records.append({
            "rank": rank, "mode": mode, "communication_events": len(comm), "copy_events": len(copies),
            "communication_event_ms": sum(b - a for a, b in comm_intervals) / 1000,
            "copy_kernel_ms": copy_time / 1000, "overlap_ms": overlap / 1000,
            "copy_time_overlapped_fraction": overlap / copy_time,
            "kernel_csv": str(paths[0].relative_to(root)),
        })
result = {"records": records, "interpretation": "Interval overlap of HCCL communication events and TensorMove device kernels; communication events may include waiting. These profiled samples are excluded from timing statistics."}
(root / "profile-summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
