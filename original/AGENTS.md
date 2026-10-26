# Project review instructions

## Versioned review record

After every generated dataset or release version (`vN`), create or update a
matching root-level review file named `feedback_vN.md`.

Every versioned feedback file must identify:

- the dataset/release version;
- the reviewer name;
- the exact reviewing model or model family, when available;
- the review date and timezone;
- the artifacts and output inspected;
- the verdict and all material feedback, including release blockers, validation
  gaps, accepted limitations, and required remediation.

Do not treat a generator claim or a built-in `PASS` as independently verified.
Clearly distinguish code inspection, test results, and measurements taken from
the completed release. If no independent review was performed, record that fact
instead of omitting the model and feedback fields.

## Feedback lives in a markdown document

Keep all review feedback in the versioned markdown review file
(`feedback_vN.md`) at the project root — not only in the chat response. The chat
response is a short summary that points at the file; the file is the permanent
record and must contain the complete verdict and every material finding.

Record in the feedback file the exact model identity you are running under
(model name plus provider/model ID, when available), alongside the reviewer
name. When adding feedback to an existing `feedback_vN.md`, update the header
provenance (model, date, artifacts, evidence basis) rather than appending an
anonymous addendum.

Never delete, overwrite, or shorten feedback from another model or reviewer.
For each additional reviewer, append a separately attributed section containing
the reviewer/model identity, review date, evidence basis, verdict, and complete
feedback. Preserve all earlier review sections verbatim.

## Session hand-off

At the end of every working session, create or update a root-level
`hand_off_vN.md` for the version the session produced. If the session produced no
new dataset, create the hand-off for the next planned version and mark it
`planned` — do not invent a release.

Every hand-off must state:

- the reviewer or hand-off author name and the reviewing model identity;
- its status explicitly: `planned`, `in progress`, `released`, or `rejected`;
- the currently released version and that release's measured gate result;
- what changed since the previous hand-off, with the evidence tier
  (measurement, test, or code inspection);
- open scope, each item with an acceptance test;
- the non-regression contract (the invariants that must stay at zero);
- the matching `feedback_vN.md`, `release_manifest.json` and validation report;
- the build and release commands.

A hand-off may only claim `released` when the completed artifact exists and:

- the build exited `0`;
- `release_manifest.json` reports `"released": true`;
- the validation report has zero `FAIL`;
- the independent checks (not the project validator) reproduce;
- every measured figure in the hand-off comes from that artifact, with no
  placeholders or carried-over numbers.

Keep earlier hand-offs on disk. When a hand-off is superseded, add a
`SUPERSEDED` banner at the top and preserve the original content; never delete or
shorten it.

After reviewing `hand_off_vN.md`, if `hand_off_v(N+1).md` does not exist, create
it as a `planned` hand-off. Include the reviewer name and model identity, and do
not invent a build, manifest, measurements, or release status for the next
version.


