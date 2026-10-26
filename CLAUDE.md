# Project instructions

## Review feedback attribution

This project collects review feedback from multiple AI models across build
iterations (see `docs/archive/feedback.md`, `docs/archive/feedback_v5.md`, `docs/archive/feedback_v6.md`, and
`hand_off_v*.md`'s "Reviews answered" line). Every `feedback_*.md` file must open
with a header block identifying which model actually produced it:

```markdown
**Reviewer:** <model name, e.g. "Claude (Sonnet 5)", "Codex", "Kimi", "MiMo">
**Review status:** ...
**Review time:** ...
**Scope:** ...
**Modification policy:** ...
```

Rules:

- The `Reviewer` line must name the model that actually generated the feedback —
  never a different model's name, even if asked, since downstream hand-off
  documents (`hand_off_v*.md`) track findings per-model and mislabeling breaks
  that ledger.
- Each new review round gets its own `feedback_v{N}.md` file (matching the build
  version it reviewed), not an edit to a prior version's file.
- When a hand-off document lists "Reviews answered," cross-check the model names
  there against the actual `feedback_*.md` files before claiming a finding is
  resolved.
