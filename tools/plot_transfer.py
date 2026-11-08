"""Plot measured temporary allocation and transfer/copy-loader latency."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parents[1] / "experiments/bounded-transfer-20261102"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), constrained_layout=True)
colors = {"A": "#68717d", "B": "#0072b2", "C": "#009e73"}
labels = {"A": "A: legacy (2 slots)", "B": "B: bounded (1 slot)", "C": "C: bounded (2 slots)"}
buckets = [128, 512, 2048]
for row, inventory in enumerate((4, 16)):
    for index, mode in enumerate(("A", "B", "C")):
        data = [json.loads((root / f"matrix-i{inventory}-b{bucket}-l2048/summary.json").read_text())["modes"][mode] for bucket in buckets]
        positions = np.arange(3) + (index - 1) * 0.23
        memory = [cell["receiver_temporary_allocated_bytes"]["median"] / 2**30 for cell in data]
        medians = [cell["complete_transfer_and_copy_loader_seconds"]["median"] for cell in data]
        p95 = [cell["complete_transfer_and_copy_loader_seconds"]["p95"] for cell in data]
        axes[row, 0].bar(positions, memory, width=0.22, color=colors[mode], label=labels[mode])
        axes[row, 1].bar(positions, medians, width=0.22, color=colors[mode], label=labels[mode])
        axes[row, 1].scatter(positions, p95, color=colors[mode], marker="_", s=130, linewidths=2)
    for column in range(2):
        ax = axes[row, column]
        ax.set_xticks(range(3), [str(bucket) for bucket in buckets])
        ax.set_xlabel("Bucket capacity (MiB)")
        ax.set_title(f"{inventory} GiB weight inventory; 2 GiB largest parameter", loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    axes[row, 0].set_ylabel("Additional allocated HBM (GiB)")
    axes[row, 1].set_ylabel("Transfer + copy loader (seconds)")
axes[0, 0].legend(frameon=False, fontsize=8)
axes[0, 1].set_ylim(0, axes[0, 1].get_ylim()[1] * 1.2)
axes[0, 1].text(0.98, 0.98, "Bars: median; ticks: p95\n30 paired samples per variant", transform=axes[0, 1].transAxes, ha="right", va="top", fontsize=8)
fig.suptitle("Bounded large-parameter HCCL transfer on two Ascend 910B2C NPUs", fontsize=13)
for extension in ("pdf", "svg", "png"):
    fig.savefig(root / f"memory-latency.{extension}", dpi=180)

fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), constrained_layout=True)
native_names = ["A", "A cached", "B", "C"]
native_colors = [colors["A"], "#a6adb5", colors["B"], colors["C"]]
for index, (name, color) in enumerate(zip(native_names, native_colors)):
    directory = "native-legacy-cached" if name == "A cached" else f"native-large-tp1-{name}"
    data = json.loads((root / directory / "update-tp1.json").read_text())
    samples = data["worker_memory"]
    peak = max(row["after"][0]["peak_allocated"] - row["before"][0]["allocated"] for row in samples[3:]) / 2**30
    times = np.array(data["complete_update_seconds"][3:]) * 1000
    axes[0, 0].bar(index, peak, color=color)
    q25, median, q75 = np.percentile(times, [25, 50, 75])
    axes[0, 1].errorbar(index, median, yerr=[[median - q25], [q75 - median]],
                       fmt="o", color=color, capsize=4, markersize=6)
    axes[0, 1].scatter(index, np.percentile(times, 95), color=color, marker="_", s=160, linewidths=2)
    axes[1, 0].plot(range(1, len(samples) + 1), [row["after"][0]["reserved"] / 2**30 for row in samples], label=name, color=color)
for column in (0, 1):
    axes[0, column].set_xticks(range(4), native_names)
axes[0, 0].set_ylabel("Worker additional allocated HBM (GiB)")
axes[0, 0].set_title("Includes native layer reconstruction", fontsize=10, loc="left")
axes[0, 1].set_yscale("log")
axes[0, 1].set_ylabel("Complete native update (ms, log scale)")
axes[0, 1].set_title("Median / IQR points; p95 ticks; 30 samples", fontsize=10, loc="left")
axes[1, 0].set(xlabel="Update", ylabel="Worker reserved HBM (GiB)")
axes[1, 0].legend(frameon=False, ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.01))
stability = json.loads((root / "native-stability/update-tp1.json").read_text())
for field, color in (("allocated", colors["B"]), ("reserved", colors["C"])):
    axes[1, 1].plot(range(1, len(stability["worker_memory"]) + 1), [row["after"][0][field] / 2**30 for row in stability["worker_memory"]], label=field, color=color)
axes[1, 1].set(xlabel="Update", ylabel="Worker HBM after update (GiB)")
axes[1, 1].set_title(f"C: {len(stability['worker_memory'])} updates over {stability['stability_elapsed_seconds'] / 60:.2f} minutes", fontsize=10, loc="left")
axes[1, 1].legend(frameon=False, fontsize=8)
for ax in axes.flat:
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
fig.suptitle("Native Qwen2 reload: 4.53 GiB weights, 2 GiB largest parameter", fontsize=13)
for extension in ("pdf", "svg", "png"):
    fig.savefig(root / f"native-evidence.{extension}", dpi=180)
