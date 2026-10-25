$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifest = Get-Content -Raw (Join-Path $root "upstreams.json") | ConvertFrom-Json

$materials = @(
    @{
        Name = "source archive"
        Path = Join-Path $root $manifest.source.archive_local_path
        Sha256 = $manifest.source.archive_sha256
    },
    @{
        Name = "GPT-2 Small paired reference archive"
        Path = Join-Path $root $manifest.references[0].archive_local_path
        Sha256 = $manifest.references[0].archive_sha256
    },
    @{
        Name = "secondary reference archive"
        Path = Join-Path $root $manifest.references[1].archive_local_path
        Sha256 = $manifest.references[1].archive_sha256
    },
    @{
        Name = "GPT-2 Small/base checkpoint"
        Path = Join-Path $root $manifest.checkpoint.local_path
        Sha256 = $manifest.checkpoint.sha256
        Size = [int64]$manifest.checkpoint.size_bytes
    }
)

foreach ($material in $materials) {
    if (-not (Test-Path -LiteralPath $material.Path -PathType Leaf)) {
        throw "$($material.Name) is missing: $($material.Path)"
    }
    $file = Get-Item -LiteralPath $material.Path
    if ($material.ContainsKey("Size") -and $file.Length -ne $material.Size) {
        throw "$($material.Name) size mismatch: expected $($material.Size), got $($file.Length)"
    }
    $actual = (Get-FileHash -LiteralPath $material.Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $material.Sha256) {
        throw "$($material.Name) SHA-256 mismatch: expected $($material.Sha256), got $actual"
    }
    Write-Output "OK  $($material.Name)"
}

$checkouts = @(
    Join-Path $root $manifest.source.local_path
    Join-Path $root $manifest.references[0].local_path
    Join-Path $root $manifest.references[1].local_path
)
foreach ($checkout in $checkouts) {
    if (-not (Test-Path -LiteralPath $checkout -PathType Container)) {
        throw "checkout is missing: $checkout"
    }
    Write-Output "OK  checkout $checkout"
}
