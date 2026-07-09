param(
    [string]$DatasetRoot = "",
    [string]$CandidatesCsv = "",
    [string]$OutputHtml = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $CandidatesCsv) {
    $CandidatesCsv = Join-Path $repoRoot "outputs\early_profile\face_signing_ahash_candidates.csv"
}
if (-not $OutputHtml) {
    $OutputHtml = Join-Path $repoRoot "outputs\early_profile\face_signing_review.html"
}
if (-not $DatasetRoot) {
    $outerDatasetDir = Get-ChildItem -Path $repoRoot -Directory |
        Where-Object { $_.Name -like "23-*" } |
        Select-Object -First 1
    if (-not $outerDatasetDir) {
        throw "Could not auto-detect dataset directory. Pass -DatasetRoot explicitly."
    }
    $innerDatasetDir = Get-ChildItem -Path $outerDatasetDir.FullName -Directory |
        Select-Object -First 1
    if (-not $innerDatasetDir) {
        throw "Could not auto-detect inner dataset directory. Pass -DatasetRoot explicitly."
    }
    $DatasetRoot = $innerDatasetDir.FullName
}

function Encode-Html {
    param([string]$Value)
    return [System.Net.WebUtility]::HtmlEncode($Value)
}

function Get-FileUri {
    param([string]$Path)
    return ([System.Uri]::new((Resolve-Path -LiteralPath $Path).Path)).AbsoluteUri
}

$candidates = @(Import-Csv -LiteralPath $CandidatesCsv | Sort-Object {
    [int]$_.ahash_hamming_distance
})
$cards = New-Object System.Text.StringBuilder

for ($i = 0; $i -lt $candidates.Count; $i++) {
    $item = $candidates[$i]
    $leftPath = Join-Path $DatasetRoot $item.left_path
    $rightPath = Join-Path $DatasetRoot $item.right_path
    $leftUri = Get-FileUri $leftPath
    $rightUri = Get-FileUri $rightPath
    $pairId = "{0}__{1}" -f $item.left_loan_id, $item.right_loan_id
    $rank = $i + 1

    [void]$cards.AppendLine(@"
      <article class="pair" data-pair="$(Encode-Html $pairId)">
        <header>
          <div><span class="rank">#$rank</span><strong>$(Encode-Html $item.left_loan_id)</strong><span class="arrow">vs</span><strong>$(Encode-Html $item.right_loan_id)</strong></div>
          <span class="distance">aHash distance: $(Encode-Html $item.ahash_hamming_distance)</span>
        </header>
        <div class="images">
          <figure><img src="$(Encode-Html $leftUri)" alt="$(Encode-Html $item.left_loan_id)"><figcaption>$(Encode-Html $item.left_loan_id)</figcaption></figure>
          <figure><img src="$(Encode-Html $rightUri)" alt="$(Encode-Html $item.right_loan_id)"><figcaption>$(Encode-Html $item.right_loan_id)</figcaption></figure>
        </div>
        <div class="actions" role="group" aria-label="Review result">
          <button data-value="similar">Likely similar</button>
          <button data-value="different">Different</button>
          <button data-value="uncertain">Uncertain</button>
        </div>
      </article>
"@)
}

$generatedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$html = @"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Face Signing Candidate Review</title>
  <style>
    :root { color-scheme: light; font-family: Arial, sans-serif; color: #17202a; background: #f4f6f8; }
    * { box-sizing: border-box; }
    body { margin: 0; }
    .topbar { position: sticky; top: 0; z-index: 2; padding: 16px 24px; background: #fff; border-bottom: 1px solid #d9dee5; }
    .topbar h1 { margin: 0 0 6px; font-size: 22px; }
    .topbar p { margin: 0; color: #5b6573; font-size: 14px; }
    .summary { margin-top: 10px; display: flex; gap: 18px; font-size: 14px; font-weight: 700; }
    main { width: min(1180px, calc(100% - 32px)); margin: 20px auto 48px; display: grid; gap: 16px; }
    .pair { background: #fff; border: 1px solid #d9dee5; border-radius: 8px; overflow: hidden; }
    .pair header { min-height: 52px; padding: 12px 16px; display: flex; align-items: center; justify-content: space-between; gap: 16px; border-bottom: 1px solid #e5e8ec; }
    .rank { display: inline-block; min-width: 42px; color: #687483; }
    .arrow { margin: 0 10px; color: #687483; }
    .distance { white-space: nowrap; padding: 5px 8px; background: #eef2f5; border-radius: 4px; font: 13px Consolas, monospace; }
    .images { display: grid; grid-template-columns: 1fr 1fr; gap: 2px; background: #d9dee5; }
    figure { margin: 0; position: relative; background: #101418; }
    img { width: 100%; aspect-ratio: 1 / 1; object-fit: contain; display: block; }
    figcaption { position: absolute; left: 8px; bottom: 8px; padding: 4px 7px; background: rgba(0,0,0,.72); color: #fff; font-size: 13px; border-radius: 4px; }
    .actions { padding: 12px 16px; display: flex; gap: 8px; }
    button { min-height: 36px; padding: 0 13px; border: 1px solid #b8c0ca; border-radius: 6px; background: #fff; cursor: pointer; font-weight: 700; }
    button:hover { background: #f1f4f6; }
    button.selected { color: #fff; border-color: #1769aa; background: #1769aa; }
    @media (max-width: 680px) {
      .topbar { padding: 14px 16px; }
      main { width: calc(100% - 16px); margin-top: 8px; }
      .pair header { align-items: flex-start; flex-direction: column; gap: 7px; }
      .images { grid-template-columns: 1fr; }
      .actions { flex-wrap: wrap; }
    }
  </style>
</head>
<body>
  <section class="topbar">
    <h1>Face Signing Candidate Review</h1>
    <p>Generated $generatedAt. Lower aHash distance means more similar global appearance; it is not proof of the same person.</p>
    <div class="summary"><span>Total: $($candidates.Count)</span><span id="reviewed">Reviewed: 0</span></div>
  </section>
  <main>
$($cards.ToString())
  </main>
  <script>
    const storageKey = "face-signing-review-v1";
    const state = JSON.parse(localStorage.getItem(storageKey) || "{}");
    const cards = [...document.querySelectorAll(".pair")];
    function render() {
      cards.forEach(card => {
        const value = state[card.dataset.pair];
        card.querySelectorAll("button").forEach(button => {
          button.classList.toggle("selected", button.dataset.value === value);
        });
      });
      document.getElementById("reviewed").textContent =
        "Reviewed: " + Object.keys(state).filter(key => state[key]).length;
    }
    cards.forEach(card => card.querySelectorAll("button").forEach(button => {
      button.addEventListener("click", () => {
        state[card.dataset.pair] = button.dataset.value;
        localStorage.setItem(storageKey, JSON.stringify(state));
        render();
      });
    }));
    render();
  </script>
</body>
</html>
"@

$outputDirectory = Split-Path -Parent $OutputHtml
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
[System.IO.File]::WriteAllText($OutputHtml, $html, [System.Text.UTF8Encoding]::new($false))
Write-Host "Wrote review report to $OutputHtml"
Write-Host "Candidate pairs: $($candidates.Count)"
