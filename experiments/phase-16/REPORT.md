# Phase 16 Mac implementation checkpoint

**Phase 16 is incomplete pending actual RTX 4070 SUPER evidence.** The owner
explicitly authorized Mac implementation and a self-contained separate Windows
Work handoff. No authenticated CUDA connection exists from this Mac. No final
training configuration is selected; Phase 17 has not begun.

The native trainer now supports explicit CUDA BF16 autocast with float32 weights,
optimizer moments and strict precision-preserving recovery. CPU/MPS float32 behavior
remains the default. BF16 rejects unavailable or emulated hardware and does not
silently fall back. Float16, adapter mixed precision, activation checkpointing,
fused optimizers and compilation are outside this bounded first protocol.

The verified Data 2.0 probe reader coalesces retained documents without crossing
source-document boundaries, preserves adjacent chunks, and uses overlap-one windows.
Hash selection and bounded omissions are recorded before candidate scoring. This is
a bounded in-memory probe adapter, not a full-corpus Phase 17 training commitment.

Actual Mac input preparation selected 4,096 training documents (1,938,919 content
IDs, 1,943,015 targets, 6,105 windows) and 128 development documents (56,908 content
IDs, 57,036 targets, 184 windows). No documents were omitted. At context 512,
training fixed-slot utilization rises from 22.14% for isolated records to 62.28%
for coalesced documents on the same selected material. Document seams and newline
joins change token counts. These are data-layout measurements, not CUDA speedups.
See [input accounting](input-check.json); full selection identities remain local.

The [prespecified plan](PLAN.md) compares 34,087,424 / 53,361,152 / 72,634,880 native
parameters using common data and exposure rules. The independent comparison varies
depth with other dimensions fixed; none is selected. It specifies numerical
controls, float32/BF16 runs, synchronized throughput, peak allocated/reserved VRAM,
host memory, initial/midpoint/final checkpoint costs, recovery and inference targets.
All attempts and imported source snapshots are retained. No accepted Phase 17 base
is produced and no final held-out evaluation is read by the runner.

Read [Mac validation](validation.json), [separate review](REVIEW.md) and the
[Windows handoff](../../docs/PHASE16_WINDOWS_HANDOFF.md). Mac CPU/MPS unit mechanics,
original synthetic controls, real data layout and future CUDA learned feasibility
are separate evidence. Phase 14 quality gates and historical negative results
remain unchanged. No new paid resources, dependency upgrades, remote writes, tags,
releases or asset uploads. Package version remains 1.3.0; the existing lock is retained.

Next action: run the reviewed transfer package in a separate Windows Work session,
retain all results and failures, and return evidence to the authoritative Mac.
Review actual CUDA results before choosing a final configuration and Phase 17
budget, closing Phase 16 or presenting its normal explicit publication checkpoint.
