# Separate Phase 12 closure review

Documentation/evidence review performed after drafting within this task. No new
runtime suite, research run, training, benchmark or build was needed or run. No
Windows/CUDA/physical Linux execution by this Mac session is implied.

Review resolutions:

- Attribute corrected Windows 301 passes / 13 expected skips in both fresh checkout
  and fresh non-editable wheel to the owner's independent summary. Native inventory
  roundtrip and strict missing/extra/corrupt rejection pass. Preserve the previous
  failed Windows run and its 299/13 suite; it stopped before actual Phase 12 CUDA.
- Distinguish Windows's fixed random synthetic 512-capacity fixture from unavailable
  pinned learned Mac models/tokenizer. Windows removes 16 **initial** blocks (zero
  replies → one); Mac removes 44 **learned retry** blocks. Do not count the synthetic
  exercise or scripted retries as reproduction of Mac learned retry transitions.
- Preserve zero learned successes across distinct scopes: Windows 1,024 CUDA tokens,
  0/16 JSON, no calls, 0/12 tasks and 0/4 failures handled; original Mac per-model
  denominators 41/64/64 at 512 and unchanged tokens/results. Scripted 12/12 and 4/4
  are mechanics. H1 passing does not convert H2 failure into capability improvement.
- Keep actual Windows CUDA coverage bounded: 512 positions, 513/over-budget rejection,
  no truncation, reservation boundaries and cache atol=rtol=1e-5. Attribute approximate
  maximum absolute difference 1.78814e-7 to the owner. Do not infer general determinism,
  cross-device equality, useful instruction following, safety or production performance.
- Do not invent raw Windows logs/hashes, external harness/prompt/tokenizer identities,
  individual skips or new driver/runtime versions. Keep owner-reported 232/3,129 file
  preservation and zero CUDA reserved-payload access attempts distinct from local audit.
- Inspect exact correction run 35880051473/job 107245869446: 303 CPU passes / 11 MPS
  skips; Phase 12 eight CPU passes and one skip. Inventory variants are native POSIX
  and simulated Windows on Linux. Tiny CPU generation and scripted retries are not
  the Windows 16-session CUDA exercise or full learned Mac experiment. No fresh wheel
  suite, full artifact/guard audit or physical Linux is implied by workflow builds.
- Keep independent dense CPU fixture recovery (400 updates / 131,600 targets, update-97
  resume across 303 updates) separate from tool learning and adapter/DPO resume.
- Preserve original Mac research, correction code/tests/docs and all prior evidence,
  including old Windows failure. Rehash 78,433 local files; reserved tests are opaque
  hashes only. Strict historical source-identity rules and all capability limits remain.
- Prepare Phase 13 handoff only: no extension selection or implementation. Require
  exact closure publication/main/tag verification, then an explicit fresh-chat start.
  Do not substitute the correction commit for the closure commit or bypass the gate.

Structured counts/provenance, CI metadata/log/workflow identities, original file
identity, preservation hashes, JSON, links, full staged diff and private-content/history
checks pass before commit. No actionable finding remains in this bounded closure.
Stop at main-only closure publication approval; no tag, release, assets or next experiment.
