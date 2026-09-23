# LatoS — Working instructions

Build an original small language model project in reviewable phases.

## Session context

Read PROJECT_STATE.md and ROADMAP.md when available. If the owner has supplied
`.private/CODEX_WORK_CONTEXT.md`, explicitly read that local file and the private
blueprint it points to. An ignored file may not appear in normal repository search.
Private context is local planning input and must not be copied into tracked files,
commit messages, pull requests, issues, release notes, or other GitHub metadata.
If private context is absent, use the public phase state and request the owner's
brief only when needed; do not invent missing goals or retrieve private references
from somewhere else.

## Implementation

- Write original code, documentation, tests, configuration, and assets.
- Use English throughout the project.
- Use supported stable dependencies, verify compatibility, and commit the lockfile.
- Work owns development across the roadmap, one phase at a time. Make engineering
  choices, implement, test, review, fix findings, and prepare commits autonomously.
  Do not ask the owner to run commands, coordinate reviews, choose routine technical
  details, or authorize the start of each phase.
- Record actual validation results. Do not claim tests, training, or pushes ran
  when they did not.
- Preserve attribution and provenance for dependencies, research, and data used.
- Keep a concise PROJECT_STATE.md with the active phase, evidence, and next step.

## Git and publication

- Keep `.private/` untracked. Never use `git add -f` on ignored planning content.
- Keep acquired datasets, weights, logs, credentials, and local environments out
  of normal Git history; preserve required artifacts in an appropriate destination.
- Check `git status --short`, `git diff --cached --name-only`, and the full staged
  diff before committing. Inspect filenames, commit text, and publication text
  for private planning content and identifiers supplied in the private brief.
- `.gitignore` does not remove previously tracked content and is not access control.
  If private material is tracked or already in history, report the finding and
  prepare an appropriate fix. Do not rewrite published history without authorization.
- Use explicit file paths when staging. Preserve unrelated user changes.
- Obtain one explicit push approval after each completed phase. Until then, keep
  all phase work local: no remote branch backups, PRs, tags, releases, or other
  GitHub writes. Prepare the actual reviewed commit before requesting approval.
- Present a brief outcome, validation results, limitations, exact commit, remote,
  destination branch, and proposed tag (if any), then ask: "Push Phase N to GitHub?"
  A yes authorizes only the described phase publication, not later phases' pushes.
- After approval, publish that exact state, verify the remote commit, and continue
  directly with the next phase only when no owner-imposed transition gate or pause
  is active. Honor the current gate in PROJECT_STATE.md; push approval alone never
  overrides it. Do not add routine phase-start approvals where none are required.
  If approval is declined, preserve the local result and await direction.
- If the reviewed source changes materially after approval, prepare a new summary
  and obtain approval for that changed publication. Never silently substitute it.
- Inspect any existing remote; do not guess its destination or change visibility.
  Include any necessary new-repository setup in the first phase's concrete push
  proposal. Default a new repository proposal to private. Do not write remotely
  before that proposal is approved.
- Never upload a whole local-folder archive to GitHub. Release archives must be
  assembled from reviewed tracked content and inspected before publishing.

## Execution and continuity

Continue from the active phase in PROJECT_STATE.md using the per-phase push checkpoint.
Honor explicit owner transition gates; publication approval alone does not lift them.
Use available execution tools and existing resources. The default new paid-service
budget is zero; do not rent compute, incur new charges, or change access controls.
Prefer a smaller runnable configuration over handing execution tasks to the owner.
Record unsupported hardware paths as untested. A genuine access or resource blocker
must be reported honestly; it is not permission to fabricate phase completion.

Perform a separate review pass after implementation and fix actionable findings
before presenting the push checkpoint. Manage this review within Work; do not make
the owner open another chat or relay prompts between tools. Do not require delegation
when no delegation capability is available.

Record completed work, validation, pending publication details, approval status,
and the next action in PROJECT_STATE.md. Never store reference-project identifiers
in that public file. Preserve local work across supported session handoffs; do not
promise continued execution when the Work session is stopped or unavailable.

Phases 0–12 established the implemented foundation and retained experiments.
Roadmap 2.0 defines the next bounded sequence; mechanisms do not establish learned
capability. Preserve negative results and distinguish learned-model experiments,
synthetic/scripted checks and platform validation.
