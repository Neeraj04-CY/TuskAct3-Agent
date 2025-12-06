Param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root ".venv"

if (-not (Test-Path $Venv)) {
    Write-Host "[quick-demo] Creating virtualenv at $Venv"
    & $Python -m venv $Venv
}

& "$Venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip | Out-Null
pip install -r "$Root\requirements.txt" | Out-Null
python -m playwright install | Out-Null

$env:EIKON_ALLOW_EXTERNAL = "0"
$env:EIKON_ALLOW_SENSITIVE = "0"
$env:PLAYWRIGHT_BYPASS_DRY_RUN = "0"
New-Item -ItemType Directory -Path "$Root\tmp_run" -Force | Out-Null
python "$Root\run_autonomy_demo.py" --summary "$Root\tmp_run\quick_autonomy.json"

$artifactRoot = Join-Path $Root "artifacts"
$latest = Get-ChildItem $artifactRoot -Directory -Filter "autonomy_demo_*" | Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $latest) {
    throw "[quick-demo] Unable to locate artifacts/autonomy_demo_*"
}
$target = Join-Path $Root "docs/artifacts/heroku_sample"
if (Test-Path $target) {
    Remove-Item $target -Recurse -Force
}
New-Item -ItemType Directory -Path (Split-Path $target -Parent) -Force | Out-Null
Copy-Item $latest.FullName $target -Recurse
python "$Root\scripts\make_demo_gif.py" --run "$target" --output "$Root\docs\assets\demo.gif" --fps 2
python "$Root\scripts\generate_run_summary.py" --run "$target" --title "Autonomy Demo (dry-run)"

Write-Host "[quick-demo] Refreshed docs/artifacts/heroku_sample and docs/assets/demo.gif"
