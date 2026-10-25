param(
    [int]$Version = 8
)

# Build a release end-to-end and scaffold the review file only if the gate passes.
#
# Usage:
#   .\run_release.ps1            # builds v8
#   .\run_release.ps1 -Version 9 # builds v9
#
# To run it unattended with Windows Task Scheduler:
#   schtasks /Create /SC ONCE /ST 02:00 /TN "CC-Release-v8" `
#     /TR "powershell -ExecutionPolicy Bypass -File \"$PWD\run_release.ps1\" -Version 8"
#
# Note: scheduling the build is possible; scheduling an *agent review* is not.
# The feedback scaffold is created with an explicit "pending review" marker.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

$out = "credit_card_data_v$Version"
$log = "build_v" + $Version + "_log.txt"
$env:CC_OUT_DIR = $out

Write-Host "Building $out ..."
python -m build.run_all *> $log
$code = $LASTEXITCODE

if ($code -ne 0) {
    Write-Host "BUILD FAILED (exit $code). No feedback file written. See $log"
    exit $code
}

$manifestPath = Join-Path $out "release_manifest.json"
if (-not (Test-Path $manifestPath)) {
    Write-Host "MANIFEST MISSING. Treating as failure."
    exit 1
}

$m = Get-Content $manifestPath -Raw | ConvertFrom-Json
if (-not $m.released) {
    Write-Host "NOT RELEASED: validation_failures=$($m.validation_failures). No feedback file written."
    exit 1
}

$fb = "feedback_v$Version.md"
if (-not (Test-Path $fb)) {
    $template = @'
# Feedback - dataset vN

> **Status: PENDING INDEPENDENT REVIEW.**
> This file was scaffolded by the release gate. No model has reviewed the
> artifact yet. The provenance fields below must be completed by the first
> reviewer; later reviewers append their own section and must not edit this one.

## Release gate (measured)

| Field | Value |
|---|---|
| Version | vN |
| Released | true |
| Validation failures | 0 |
| Manifest | credit_card_data_vN/release_manifest.json |

## Reviewer provenance - REQUIRED (first reviewer)

| Field | Value |
|---|---|
| Reviewer name | *(to be completed)* |
| Model name | *(to be completed)* |
| Provider / model ID | *(to be completed)* |
| Review date and timezone | *(to be completed)* |
| Artifacts inspected | *(to be completed)* |
| Evidence basis | measurement / test / code inspection (state which) |
| Verdict | *(to be completed)* |

## Findings

*(the first reviewer writes the complete verdict and every material finding here)*

---

*Later reviewers: append a separately attributed section below. Preserve all
earlier sections verbatim.*
'@
    $template = $template.Replace("vN", "v$Version")
    Set-Content -Path $fb -Value $template -Encoding UTF8
    Write-Host "Created $fb (pending independent review)."
}

Write-Host ""
Write-Host "v$Version released with $($m.validation_failures) validation failures."
Write-Host "Next steps:"
Write-Host "  1. Ask MiMo, Codex and Kimi to append sections to $fb"
Write-Host "  2. Each must record model name + provider/model ID, date, artifacts and evidence basis"
Write-Host "  3. Update hand_off_v$Version.md from 'planned' to 'released' with measured numbers"
exit 0
