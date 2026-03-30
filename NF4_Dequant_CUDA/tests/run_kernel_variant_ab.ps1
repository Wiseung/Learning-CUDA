param(
    [string]$WeightsBin = "tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin",
    [string]$ParamsTemplate = "params.txt",
    [string]$OutputDir = "tests/data",
    [string]$ExePath = "build/Release/nf4_dequant.exe",
    [int]$Rounds = 5
)

$ErrorActionPreference = "Stop"

function Set-OrAddParamLine {
    param(
        [string[]]$Lines,
        [string]$Key,
        [string]$Value
    )

    $updated = $false
    $result = New-Object System.Collections.Generic.List[string]
    foreach ($line in $Lines) {
        $trimmed = $line.Trim()
        if (-not $updated -and $trimmed -match "^$([regex]::Escape($Key))\s*=") {
            $result.Add("$Key = $Value")
            $updated = $true
        } else {
            $result.Add($line)
        }
    }
    if (-not $updated) {
        $result.Add("$Key = $Value")
    }
    return $result
}

function Get-PerfValue {
    param(
        [string]$PerfLogPath,
        [string]$Key
    )

    $line = Select-String -Path $PerfLogPath -Pattern "^$([regex]::Escape($Key))=" | Select-Object -First 1
    if ($null -eq $line) {
        throw "Missing key '$Key' in perf log: $PerfLogPath"
    }
    return ($line.Line.Split("=", 2)[1]).Trim()
}

function Get-Median {
    param([double[]]$Values)
    if ($Values.Count -eq 0) {
        throw "Get-Median requires at least one value."
    }
    $sorted = $Values | Sort-Object
    if (($sorted.Count % 2) -eq 1) {
        return [double]$sorted[[int]($sorted.Count / 2)]
    }
    $hi = [int]($sorted.Count / 2)
    $lo = $hi - 1
    return ([double]$sorted[$lo] + [double]$sorted[$hi]) / 2.0
}

$variants = @("generic", "specialized")
$results = @{}
foreach ($variant in $variants) {
    $results[$variant] = New-Object System.Collections.Generic.List[double]
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$templateLines = Get-Content $ParamsTemplate

for ($round = 1; $round -le $Rounds; ++$round) {
    foreach ($variant in $variants) {
        $paramsLines = $templateLines
        $paramsLines = Set-OrAddParamLine -Lines $paramsLines -Key "kernel_variant" -Value "`"$variant`""
        $paramsLines = Set-OrAddParamLine -Lines $paramsLines -Key "autotune_block_dim" -Value "false"

        $paramsPath = Join-Path $OutputDir ("params_variant_{0}_r{1}.txt" -f $variant, $round)
        $outputBin = Join-Path $OutputDir ("out_variant_{0}_r{1}.bin" -f $variant, $round)
        $perfLog = "$outputBin.perf.log"

        Set-Content -Path $paramsPath -Value $paramsLines -Encoding ASCII
        Write-Host ("[Round {0}/{1}] Running variant={2}" -f $round, $Rounds, $variant)
        & $ExePath $WeightsBin $paramsPath $outputBin
        if ($LASTEXITCODE -ne 0) {
            throw "Execution failed for variant=$variant round=$round"
        }

        $kernelMs = [double](Get-PerfValue -PerfLogPath $perfLog -Key "kernel_time_ms")
        $effectiveVariant = Get-PerfValue -PerfLogPath $perfLog -Key "kernel_variant_effective"
        Write-Host ("  kernel_time_ms={0:N6} effective={1}" -f $kernelMs, $effectiveVariant)
        $results[$variant].Add($kernelMs)
    }
}

$genericMedian = Get-Median -Values $results["generic"].ToArray()
$specializedMedian = Get-Median -Values $results["specialized"].ToArray()
$speedup = if ($specializedMedian -gt 0.0) { $genericMedian / $specializedMedian } else { [double]::NaN }
$deltaPct = if ($genericMedian -gt 0.0) { (($specializedMedian - $genericMedian) / $genericMedian) * 100.0 } else { 0.0 }

$summaryPath = Join-Path $OutputDir "kernel_variant_ab.csv"
$summary = @(
    "rounds,$Rounds"
    "metric,generic_median_ms,specialized_median_ms,delta_pct,speedup_generic_over_specialized"
    ("kernel_time_ms,{0:F6},{1:F6},{2:F2},{3:F4}" -f $genericMedian, $specializedMedian, $deltaPct, $speedup)
)
Set-Content -Path $summaryPath -Value $summary -Encoding ASCII

Write-Host "Done."
Write-Host "Summary: $summaryPath"
Get-Content $summaryPath
