# Bounded HCCL updates on Ascend 910B

Byte fragmentation reduces the measured receiver transport peak from 4.375 to
2.250 GiB (48.6%) and the native Qwen2 extra Worker peak from 6.125 to 4.250 GiB
(30.6%) for a 2 GiB parameter and 128 MiB buckets. The native update completes
1001 rounds over 1800.33 seconds with final full-byte and cold-inference agreement.

- [Experiment report](experiments/bounded-transfer-20261102/REPORT.md)
- [Environment and reproduction commands](experiments/bounded-transfer-20261102/README.txt)
- [Validation results](experiments/bounded-transfer-20261102/acceptance.json)
- [Patch for the measured source pairing](experiments/bounded-transfer-20261102/bounded-transfer.patch)
- [Submission patch for current main](experiments/bounded-transfer-20261102/submission.patch)

Raw measurements, pressure failures, profiler traces and reproduction tools are
included in this branch. The feature PR contains only implementation, tests,
user documentation and its release note.

## Revision coverage

Hardware measurements use Ascend `24a76e84394ae3427195e3715fc08e90769d398c`
and vLLM `544f182d9f0c9bfe8bd7645525951d5e3692a542`, with #19127 head
`5d30f81d0cf777168b472cfe9ed45b380ed5721c` as the common lifecycle baseline.

The submission targets Ascend `28e9e97ba254b076a88275e8a275556cb9a11664`.
Its weight-transfer CPU suite passes 214 tests with vLLM main
`24420b7a341af37a9bc3186e6fc6062512e0aa58` and 214 with v0.32.0
`5df63a7203cdb9328d824a27b682e77c7b9928be`. The two runs exercise the
respective upstream broadcast signatures. Full `bash format.sh ci` passes.
Publication validation logs are under the experiment's `logs/` directory.

## Limitations

Current-head NPU integration remains required before merge. The hardware results
apply to the fixed experiment pairing. Compared with a legacy control that only
reuses streams, bounded double buffering changes native complete-update median
by -2.6% and p95 by +23.9%, while reducing the allocated peak by 30.6%.
Quantized, MoE, multiple sending roots and 8/16-card execution are untested.

OpenAI Codex assisted with implementation, tests, experiments and PR preparation.
The submitter reviewed the changes. Local validation ran through the automated
workflow described in the experiment artifacts.
