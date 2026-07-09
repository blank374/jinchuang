param(
    [int]$BatchSize = 8,
    [int]$TopK = 5,
    [string]$Device = "auto",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

if (-not $Python) {
    $condaPython = "F:\Environment\conda_envs\pytorch\python.exe"
    $Python = if (Test-Path -LiteralPath $condaPython) { $condaPython } else { "python" }
}

& $Python -m mvp.pipeline --batch-size $BatchSize --top-k $TopK --device $Device

Write-Host ""
Write-Host "MVP completed. Start the dashboard with:"
Write-Host "$Python -m streamlit run dashboard.py"
