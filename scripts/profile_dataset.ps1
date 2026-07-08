param(
    [string]$DatasetRoot = "",
    [string]$OutputRoot = "",
    [int]$SamplesPerClass = 5,
    [int]$AHashHammingThreshold = 8
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

$repoRoot = Split-Path -Parent $PSScriptRoot

if (-not $OutputRoot) {
    $OutputRoot = Join-Path $repoRoot "outputs\early_profile"
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

function Get-RelativePath {
    param(
        [string]$BasePath,
        [string]$TargetPath
    )

    $baseUri = [System.Uri]((Resolve-Path $BasePath).Path.TrimEnd('\') + '\')
    $targetUri = [System.Uri]((Resolve-Path $TargetPath).Path)
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString()).Replace('/', '\')
}

function Get-AverageHash {
    param([string]$Path)

    $source = $null
    $small = $null
    $graphics = $null
    try {
        $source = [System.Drawing.Bitmap]::new($Path)
        $small = [System.Drawing.Bitmap]::new(8, 8)
        $graphics = [System.Drawing.Graphics]::FromImage($small)
        $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBilinear
        $graphics.DrawImage($source, 0, 0, 8, 8)

        $values = New-Object System.Collections.Generic.List[double]
        for ($y = 0; $y -lt 8; $y++) {
            for ($x = 0; $x -lt 8; $x++) {
                $pixel = $small.GetPixel($x, $y)
                $gray = (0.299 * $pixel.R) + (0.587 * $pixel.G) + (0.114 * $pixel.B)
                $values.Add($gray)
            }
        }

        $average = ($values | Measure-Object -Average).Average
        $bits = New-Object System.Text.StringBuilder
        foreach ($value in $values) {
            [void]$bits.Append($(if ($value -ge $average) { "1" } else { "0" }))
        }

        $hex = New-Object System.Text.StringBuilder
        for ($i = 0; $i -lt 64; $i += 4) {
            $nibble = $bits.ToString().Substring($i, 4)
            [void]$hex.Append([Convert]::ToInt32($nibble, 2).ToString("x"))
        }
        return $hex.ToString()
    }
    finally {
        if ($graphics) { $graphics.Dispose() }
        if ($small) { $small.Dispose() }
        if ($source) { $source.Dispose() }
    }
}

function Get-HexHammingDistance {
    param(
        [string]$Left,
        [string]$Right
    )

    if (-not $Left -or -not $Right -or $Left.Length -ne $Right.Length) {
        return $null
    }

    $distance = 0
    for ($i = 0; $i -lt $Left.Length; $i++) {
        $leftValue = [Convert]::ToInt32($Left.Substring($i, 1), 16)
        $rightValue = [Convert]::ToInt32($Right.Substring($i, 1), 16)
        $xorValue = $leftValue -bxor $rightValue
        while ($xorValue -gt 0) {
            $distance += ($xorValue -band 1)
            $xorValue = $xorValue -shr 1
        }
    }
    return $distance
}

function Get-ImageInfo {
    param([System.IO.FileInfo]$File)

    $image = $null
    try {
        $image = [System.Drawing.Image]::FromFile($File.FullName)
        $width = $image.Width
        $height = $image.Height
        $aspectRatio = if ($height -gt 0) { [math]::Round($width / $height, 4) } else { $null }
        return @{
            OpenStatus = "ok"
            Width = $width
            Height = $height
            AspectRatio = $aspectRatio
            Error = ""
        }
    }
    catch {
        return @{
            OpenStatus = "bad"
            Width = $null
            Height = $null
            AspectRatio = $null
            Error = $_.Exception.Message
        }
    }
    finally {
        if ($image) { $image.Dispose() }
    }
}

if (-not (Test-Path $DatasetRoot)) {
    throw "DatasetRoot not found: $DatasetRoot"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$sampleRoot = Join-Path $OutputRoot "samples_by_class"
New-Item -ItemType Directory -Force -Path $sampleRoot | Out-Null

$extensions = @("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
$files = foreach ($extension in $extensions) {
    Get-ChildItem -Path $DatasetRoot -Recurse -File -Filter $extension
}

$rows = New-Object System.Collections.Generic.List[object]
foreach ($file in $files) {
    $imageInfo = Get-ImageInfo -File $file
    $sha256 = ""
    $ahash = ""
    if ($imageInfo.OpenStatus -eq "ok") {
        $sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
        $ahash = Get-AverageHash -Path $file.FullName
    }

    $parentName = Split-Path -Leaf $file.DirectoryName
    $imageType = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
    $relativePath = Get-RelativePath -BasePath $DatasetRoot -TargetPath $file.FullName

    $rows.Add([PSCustomObject]@{
        loan_id = $parentName
        image_type = $imageType
        file_name = $file.Name
        relative_path = $relativePath
        size_bytes = $file.Length
        size_kb = [math]::Round($file.Length / 1KB, 2)
        width = $imageInfo.Width
        height = $imageInfo.Height
        aspect_ratio = $imageInfo.AspectRatio
        open_status = $imageInfo.OpenStatus
        error = $imageInfo.Error
        sha256 = $sha256
        ahash = $ahash
    })
}

$manifestPath = Join-Path $OutputRoot "image_manifest.csv"
$rows | Sort-Object loan_id, image_type | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $manifestPath

$summaryRows = $rows |
    Group-Object image_type |
    ForEach-Object {
        $group = $_.Group
        [PSCustomObject]@{
            image_type = $_.Name
            count = $group.Count
            ok_count = ($group | Where-Object { $_.open_status -eq "ok" }).Count
            bad_count = ($group | Where-Object { $_.open_status -ne "ok" }).Count
            total_mb = [math]::Round((($group | Measure-Object size_bytes -Sum).Sum) / 1MB, 3)
            avg_kb = [math]::Round((($group | Measure-Object size_bytes -Average).Average) / 1KB, 2)
            min_width = ($group | Where-Object width | Measure-Object width -Minimum).Minimum
            max_width = ($group | Where-Object width | Measure-Object width -Maximum).Maximum
            min_height = ($group | Where-Object height | Measure-Object height -Minimum).Minimum
            max_height = ($group | Where-Object height | Measure-Object height -Maximum).Maximum
        }
    } |
    Sort-Object image_type

$summaryRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutputRoot "class_summary.csv")

$profileRows = @(
    [PSCustomObject]@{ metric = "loan_folder_count"; value = (Get-ChildItem -Path $DatasetRoot -Directory | Measure-Object).Count }
    [PSCustomObject]@{ metric = "image_count"; value = $rows.Count }
    [PSCustomObject]@{ metric = "ok_image_count"; value = ($rows | Where-Object { $_.open_status -eq "ok" }).Count }
    [PSCustomObject]@{ metric = "bad_image_count"; value = ($rows | Where-Object { $_.open_status -ne "ok" }).Count }
    [PSCustomObject]@{ metric = "total_mb"; value = [math]::Round((($rows | Measure-Object size_bytes -Sum).Sum) / 1MB, 3) }
    [PSCustomObject]@{ metric = "avg_image_kb"; value = [math]::Round((($rows | Measure-Object size_bytes -Average).Average) / 1KB, 2) }
    [PSCustomObject]@{ metric = "image_types"; value = (($rows | Select-Object -ExpandProperty image_type -Unique | Sort-Object) -join ";") }
)
$profileRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutputRoot "data_profile.csv")

$rows |
    Where-Object { $_.open_status -ne "ok" -or $_.width -lt 100 -or $_.height -lt 100 } |
    Sort-Object loan_id, image_type |
    Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutputRoot "bad_images.csv")

$duplicateRows = New-Object System.Collections.Generic.List[object]
$shaGroups = $rows | Where-Object { $_.sha256 } | Group-Object sha256 | Where-Object { $_.Count -gt 1 }
foreach ($group in $shaGroups) {
    foreach ($item in $group.Group) {
        $duplicateRows.Add([PSCustomObject]@{
            duplicate_type = "sha256_exact"
            duplicate_key = $group.Name
            loan_id = $item.loan_id
            image_type = $item.image_type
            relative_path = $item.relative_path
        })
    }
}

$hashGroups = $rows | Where-Object { $_.ahash } | Group-Object ahash | Where-Object { $_.Count -gt 1 }
foreach ($group in $hashGroups) {
    foreach ($item in $group.Group) {
        $duplicateRows.Add([PSCustomObject]@{
            duplicate_type = "ahash_same"
            duplicate_key = $group.Name
            loan_id = $item.loan_id
            image_type = $item.image_type
            relative_path = $item.relative_path
        })
    }
}
$duplicateRows |
    Sort-Object duplicate_type, duplicate_key, loan_id, image_type |
    Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutputRoot "duplicate_hash.csv")

$faceRows = @($rows | Where-Object { $_.image_type -eq "face_signing" -and $_.ahash })
$candidateRows = New-Object System.Collections.Generic.List[object]
for ($i = 0; $i -lt $faceRows.Count; $i++) {
    for ($j = $i + 1; $j -lt $faceRows.Count; $j++) {
        $distance = Get-HexHammingDistance -Left $faceRows[$i].ahash -Right $faceRows[$j].ahash
        if ($null -ne $distance -and $distance -le $AHashHammingThreshold) {
            $candidateRows.Add([PSCustomObject]@{
                left_loan_id = $faceRows[$i].loan_id
                right_loan_id = $faceRows[$j].loan_id
                left_path = $faceRows[$i].relative_path
                right_path = $faceRows[$j].relative_path
                ahash_hamming_distance = $distance
                left_ahash = $faceRows[$i].ahash
                right_ahash = $faceRows[$j].ahash
            })
        }
    }
}
$candidateRows |
    Sort-Object ahash_hamming_distance, left_loan_id, right_loan_id |
    Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutputRoot "face_signing_ahash_candidates.csv")

foreach ($imageType in ($rows | Select-Object -ExpandProperty image_type -Unique | Sort-Object)) {
    $targetDir = Join-Path $sampleRoot $imageType
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    $rows |
        Where-Object { $_.image_type -eq $imageType -and $_.open_status -eq "ok" } |
        Select-Object -First $SamplesPerClass |
        ForEach-Object {
            $source = Join-Path $DatasetRoot $_.relative_path
            $target = Join-Path $targetDir ($_.loan_id + "_" + $_.file_name)
            Copy-Item -LiteralPath $source -Destination $target -Force
        }
}

$readmePath = Join-Path $OutputRoot "README.md"
@'
# Early Dataset Profile

Generated by `scripts/profile_dataset.ps1`.

## Files

- `data_profile.csv`: overall dataset metrics.
- `class_summary.csv`: image count, size, and dimension summary by image type.
- `image_manifest.csv`: one row per image with path, size, dimensions, SHA256, and average hash.
- `bad_images.csv`: unreadable images or very small images.
- `duplicate_hash.csv`: exact SHA256 duplicates and same average-hash groups.
- `face_signing_ahash_candidates.csv`: early near-duplicate candidates among signing photos.
- `samples_by_class/`: quick visual samples for each image type.

## Next Steps

1. Review `samples_by_class/` to confirm category quality.
2. Inspect `duplicate_hash.csv` for potential repeated or near-repeated submissions.
3. Use `image_manifest.csv` as the base manifest for classification and retrieval experiments.
'@ | Set-Content -Encoding UTF8 -Path $readmePath

Write-Host "Wrote profile to $OutputRoot"
Write-Host "Images: $($rows.Count)"
Write-Host "Bad images: $(($rows | Where-Object { $_.open_status -ne 'ok' }).Count)"
Write-Host "Duplicate rows: $($duplicateRows.Count)"
Write-Host "Face signing candidates: $($candidateRows.Count)"
