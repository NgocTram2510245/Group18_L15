$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Khong tim thay $PythonExe. Hay tao .venv va cai requirements.txt truoc."
}

& $PythonExe (Join-Path $PSScriptRoot "anova_utils.py")
if ($LASTEXITCODE -ne 0) {
    throw "ANOVA pipeline failed with exit code $LASTEXITCODE."
}
& $PythonExe -m unittest discover -s $PSScriptRoot -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) {
    throw "Unit tests failed with exit code $LASTEXITCODE."
}

$RequiredOutputs = @(
    "results\anova_results.csv",
    "results\posthoc_results.csv",
    "results\summary_results.csv",
    "results\duration_zero_analysis.csv",
    "figures\report\wilson_95ci_five_groups.png",
    "figures\report\anova_ecdf_four_features.png",
    "figures\report\games_howell_heatmaps.png",
    "figures\report\qq_log_features.png",
    "figures\report\duration_zero_wilson.png"
)

foreach ($RelativePath in $RequiredOutputs) {
    $Target = Join-Path $PSScriptRoot $RelativePath
    if (-not (Test-Path -LiteralPath $Target)) {
        throw "Thieu dau ra bat buoc: $Target"
    }
}

Write-Host "ANOVA pipeline and validation completed successfully."

