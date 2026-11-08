"""Evaluate the experiment's stated acceptance criteria from measured results."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "experiments/bounded-transfer-20261102"


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    checks = {}
    checks["standalone_cpu"] = "199 passed" in (ROOT / "logs/standalone-feature-cpu.log").read_text()
    checks["e2e"] = "4 passed" in (ROOT / "logs/bounded-e2e-driver.log").read_text()
    for tp in (1, 2):
        for mode in "ABC":
            result = read(f"native-formal-tp{tp}-{mode}/comparison-tp{tp}.json")
            checks[f"native_tp{tp}_{mode}"] = result["all_parameter_bytes"] == result["cold_inference"] == "passed"
        result = read(f"native-graph-tp{tp}-C/comparison-tp{tp}.json")
        checks[f"graph_tp{tp}"] = result["all_parameter_bytes"] == result["cold_inference"] == "passed"
        baseline = read(f"native-formal-tp{tp}-A/update-tp{tp}.json")["complete_update_seconds"][3:]
        for mode in "BC":
            measured = read(f"native-formal-tp{tp}-{mode}/update-tp{tp}.json")["complete_update_seconds"][3:]
            checks[f"native_tp{tp}_{mode}_p95"] = bool(np.percentile(measured, 95) <= np.percentile(baseline, 95) * 1.1)
    improvements = []
    for inventory, bucket, largest in [(g, b, 2048) for g in (4, 16) for b in (128, 512, 2048)] + [(4, 128, 64)]:
        name = f"matrix-i{inventory}-b{bucket}-l{largest}"
        modes = read(f"{name}/summary.json")["modes"]
        for rank in (0, 1):
            records = read(f"{name}/rank{rank}.json")["records"]
            checks[f"{name}_rank{rank}_bytes"] = all(row["all_bytes_correct"] for row in records)
            checks[f"{name}_rank{rank}_collectives"] = all(
                row["mode"] == "A" or max(row["collective_bytes"], default=0) <= bucket * 2**20 for row in records
            )
        checks[f"{name}_samples"] = all(v["complete_transfer_and_copy_loader_seconds"]["n"] == 30 for v in modes.values())
        if largest > bucket:
            improvements.append(1 - modes["C"]["receiver_temporary_allocated_bytes"]["median"] / modes["A"]["receiver_temporary_allocated_bytes"]["median"])
        if largest == 64:
            for mode in "BC":
                checks[f"ordinary_copy_loader_{mode}_p95"] = modes[mode]["complete_transfer_and_copy_loader_seconds"]["p95"] <= modes["A"]["complete_transfer_and_copy_loader_seconds"]["p95"] * 1.1
    checks["temporary_memory_reduction_20_percent"] = max(improvements) >= 0.2
    pressure = read("pressure-results.json")
    failed = {row["receiver_headroom_mib"] for row in pressure if row["mode"] == "A" and not row["success"] and row["allocation_failures"]}
    checks["same_budget_baseline_oom_candidate_ten_successes"] = any(
        row["mode"] == "C" and row["success"] and row["successful_updates"] >= 10 and row["receiver_headroom_mib"] in failed for row in pressure
    )
    pressure = read("native-pressure-results.json")
    checks["native_same_budget_oom_and_ten_successes"] = (
        any(row["mode"] == "A" and row["allocation_oom"] and not row["success"] for row in pressure)
        and any(row["mode"] == "C" and row["success"] and row["successful_updates"] == 10 for row in pressure)
        and len({row["receiver_headroom_mib"] for row in pressure}) == 1
    )
    result = read("native-pressure-C/comparison-tp1.json")
    checks["native_pressure_complete_load"] = result["all_parameter_bytes"] == result["cold_inference"] == "passed"
    for mode in "BC":
        result = read(f"native-fragmented-tp2-{mode}/comparison-tp2.json")
        checks[f"fragmented_tp2_{mode}"] = result["all_parameter_bytes"] == result["cold_inference"] == "passed"
    for mode in "ABC":
        result = read(f"native-large-tp1-{mode}/comparison-tp1.json")
        checks[f"large_native_{mode}"] = result["all_parameter_bytes"] == result["cold_inference"] == "passed"
    for rank in range(4):
        rows = read(f"bytes-four-cached/rank{rank}.json")["records"]
        checks[f"four_card_rank{rank}_bytes"] = len(rows) == 36 and all(row["correct"] for row in rows)
        rows = read(f"matrix-four/rank{rank}.json")["records"]
        checks[f"four_card_rank{rank}_matrix"] = all(row["all_bytes_correct"] for row in rows) and all(
            sum(row["mode"] == mode and row["sample"] >= 0 for row in rows) == 30 for mode in "ABC"
        )
    stability = read("native-stability/update-tp1.json")
    checks["stability_count_and_duration"] = len(stability["complete_update_seconds"]) >= 1000 and stability["stability_elapsed_seconds"] >= 1800
    result = read("native-stability/comparison-tp1.json")
    checks["stability_complete_load"] = result["all_parameter_bytes"] == result["cold_inference"] == "passed" and stability["parameter_addresses_stable"]
    profile = read("profile-summary.json")["records"]
    checks["independent_profiler"] = len(profile) == 6 and all(row["communication_events"] > 0 and row["copy_events"] > 0 for row in profile)
    output = {"passed": all(checks.values()), "checks": checks, "best_temporary_memory_reduction": max(improvements)}
    (ROOT / "acceptance.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
