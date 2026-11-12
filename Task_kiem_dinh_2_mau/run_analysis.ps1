param(
    [switch]$SkipNotebook
)

$ErrorActionPreference = "Stop"
$ModuleDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ModuleDir
$Python = Join-Path $RepoRoot ".venv312\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Không tìm thấy .venv312. Xem hướng dẫn cài đặt trong two_sample_testing/README.md."
}

Push-Location $ModuleDir
try {
    & $Python -m unittest -v test_two_sample_utils.py
    if ($LASTEXITCODE -ne 0) { throw "Synthetic-data tests failed." }
    & $Python two_sample_utils.py
    if ($LASTEXITCODE -ne 0) { throw "Statistical analysis failed." }
    if (-not $SkipNotebook) {
        & $Python execute_notebook.py
        if ($LASTEXITCODE -ne 0) { throw "Notebook execution failed." }
    }
}
finally {
    Pop-Location
}

