# Fills this release folder with the code and results from your project, leaving out data,
# checkpoints, caches and logs. Nothing in your project is changed.
#
#   1. Unzip this kit to   D:\Graph-Based\glioma-qc-blind-spots
#   2. In the terminal you normally train in (Python environment active):
#        cd D:\Graph-Based\glioma-qc-blind-spots
#        powershell -ExecutionPolicy Bypass -File .\prepare_release.ps1
#   3. Read the summary it prints, then upload the folder (see HOW_TO_RELEASE.txt).

param([string]$Project = "..\topograph_seg")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path $Project)) { throw "Project folder not found: $Project (pass -Project D:\path\to\topograph_seg)" }
$Project = (Resolve-Path $Project).Path
Write-Host "Copying from $Project" -ForegroundColor Cyan

$skipDirs = @("__pycache__", ".ipynb_checkpoints", "outputs", "data", "runs", "tb", ".git", ".venv", "env")
$skipExt  = @(".pt", ".pth", ".ckpt", ".nii", ".gz", ".npz", ".npy", ".log", ".pyc", ".zip", ".tar")
$copied = 0; $skipped = @()

function Copy-Tree($rel) {
    $src = Join-Path $Project $rel
    if (-not (Test-Path $src)) { Write-Host "  (no $rel folder)" -ForegroundColor Yellow; return }
    foreach ($f in @(Get-ChildItem $src -Recurse -File)) {
        $relPath = $f.FullName.Substring($Project.Length).TrimStart('\', '/')
        $parts = $relPath -split '[\\/]'
        if (@($parts | Where-Object { $skipDirs -contains $_ -or $_ -like "cache*" }).Count -gt 0) { continue }
        if ($skipExt -contains $f.Extension.ToLower()) { $script:skipped += $relPath; continue }
        if ($f.Length -gt 20MB) { $script:skipped += "$relPath (over 20 MB)"; continue }
        $dst = Join-Path $PSScriptRoot $relPath
        New-Item -ItemType Directory -Force (Split-Path $dst -Parent) | Out-Null
        Copy-Item $f.FullName $dst -Force
        $script:copied++
    }
}

foreach ($d in @("src", "scripts", "configs", "splits", "notebooks")) { Copy-Tree $d }

# Per-timepoint results (small CSVs; subject ids are public BraTS identifiers)
New-Item -ItemType Directory -Force "results" | Out-Null
$res = @(Get-ChildItem (Join-Path $Project "outputs") -File -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -like "test_eval_*_fold0.csv" -or $_.Name -like "uncertainty_*_fold0.csv" -or
    ($_.Name -like "region_unc_*.csv" -and $_.Name -notlike "*smoke*") })
foreach ($f in $res) { Copy-Item $f.FullName (Join-Path "results" $f.Name) -Force; $copied++ }

# Replace absolute data paths in the configs by placeholders (data/<last folder>)
$pathRe = '([A-Za-z]):[\\/](?:[^"''\r\n#]*[\\/])?([^"''\\/\r\n#]+?)(?=["''\s]|$)'
foreach ($y in @(Get-ChildItem "configs" -File -Include *.yaml, *.yml -Recurse -ErrorAction SilentlyContinue)) {
    $t = Get-Content $y.FullName -Raw
    $n = [regex]::Replace($t, $pathRe, 'data/$2')
    if ($n -ne $t) { Set-Content $y.FullName $n -NoNewline; Write-Host "  data paths replaced in configs\$($y.Name)" }
}

# Exact package versions of this environment
$old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
$req = & python -c @"
import importlib.metadata as m
pk = ['torch','monai','numpy','scipy','pandas','scikit-learn','matplotlib','PyYAML','nibabel','SimpleITK',
      'torch-geometric','torch_geometric','tensorboard','tqdm','einops','timm']
seen = set()
for p in pk:
    try:
        v = m.version(p); k = p.lower().replace('_', '-')
        if k not in seen: seen.add(k); print(f'{p}=={v}')
    except m.PackageNotFoundError: pass
"@
$ErrorActionPreference = $old
if ($req) { $req | Set-Content "requirements.txt"; Write-Host "  requirements.txt written ($(@($req).Count) packages)" }

# Final check: anything that should not be public
Write-Host ""
Write-Host "==== Summary ====" -ForegroundColor Cyan
Write-Host "Files copied: $copied"
if ($skipped.Count) { Write-Host "Left out (data, checkpoints, logs, archives): $($skipped.Count) files" }
$flag = @(Get-ChildItem . -Recurse -File -Include *.py, *.yaml, *.yml, *.json, *.txt, *.md, *.ipynb |
    Where-Object { $_.Name -ne "prepare_release.ps1" } |
    Select-String -Pattern '[A-Za-z]:[\\/](Users|Graph-Based|Datasets)|password|api[_-]?key|secret|token\s*=' -List)
if ($flag.Count) {
    Write-Host "Please look at these lines before uploading (personal paths or secrets):" -ForegroundColor Yellow
    $flag | ForEach-Object { Write-Host ("  {0}:{1}: {2}" -f $_.Path.Substring($PSScriptRoot.Length + 1), $_.LineNumber, $_.Line.Trim()) }
} else {
    Write-Host "No personal paths or secrets found." -ForegroundColor Green
}
$big = @(Get-ChildItem . -Recurse -File | Where-Object { $_.Length -gt 5MB })
if ($big.Count) { Write-Host "Large files (over 5 MB):" -ForegroundColor Yellow; $big | ForEach-Object { Write-Host "  $($_.FullName)" } }
$total = (Get-ChildItem . -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Release folder size: {0:N1} MB. Next: HOW_TO_RELEASE.txt, step 3." -f $total) -ForegroundColor Green
