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

function Add-Result {
    param(
        [hashtable]$Bucket,
        [string]$Mode,
        [string]$Metric,
        [double]$Value
    )

    if (-not $Bucket.ContainsKey($Mode)) {
        throw "Unknown mode '$Mode'."
    }
    if (-not $Bucket[$Mode].ContainsKey($Metric)) {
        throw "Unknown metric '$Metric'."
    }
    $Bucket[$Mode][$Metric].Add($Value)
}

$modes = @("false", "true")
$metricKeys = @("kernel_time_ms", "end_to_end_ms", "graph_build_time_ms")
$results = @{}
foreach ($mode in $modes) {
    $results[$mode] = @{}
    foreach ($metricKey in $metricKeys) {
        $results[$mode][$metricKey] = New-Object System.Collections.Generic.List[double]
    }
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$templateLines = Get-Content $ParamsTemplate

for ($round = 1; $round -le $Rounds; ++$round) {
    $modesThisRound = if (($round % 2) -eq 1) { @("false", "true") } else { @("true", "false") }
    foreach ($mode in $modesThisRound) {
        $paramsLines = $templateLines
        $paramsLines = Set-OrAddParamLine -Lines $paramsLines -Key "use_pinned_host_input" -Value $mode
        $paramsLines = Set-OrAddParamLine -Lines $paramsLines -Key "use_cuda_graph" -Value "false"
        $paramsLines = Set-OrAddParamLine -Lines $paramsLines -Key "profile_loop_iters" -Value "1"
        $paramsLines = Set-OrAddParamLine -Lines $paramsLines -Key "kernel_warmup_iters" -Value "0"
        $paramsLines = Set-OrAddParamLine -Lines $paramsLines -Key "autotune_block_dim" -Value "false"

        $tag = if ($mode -eq "true") { "input_pinned" } else { "input_pageable" }
        $paramsPath = Join-Path $OutputDir ("params_{0}_r{1}.txt" -f $tag, $round)
        $outputBin = Join-Path $OutputDir ("out_{0}_r{1}.bin" -f $tag, $round)
        $perfLog = "$outputBin.perf.log"

        Set-Content -Path $paramsPath -Value $paramsLines -Encoding ASCII
        Write-Host ("[Round {0}/{1}] Running use_pinned_host_input={2}" -f $round, $Rounds, $mode)
        & $ExePath $WeightsBin $paramsPath $outputBin
        if ($LASTEXITCODE -ne 0) {
            throw "Execution failed for use_pinned_host_input=$mode round=$round"
        }

        $kernelMs = [double](Get-PerfValue -PerfLogPath $perfLog -Key "kernel_time_ms")
        $endToEndMs = [double](Get-PerfValue -PerfLogPath $perfLog -Key "end_to_end_ms")
        $graphBuildMs = [double](Get-PerfValue -PerfLogPath $perfLog -Key "graph_build_time_ms")
        $pinnedBuffers = [int](Get-PerfValue -PerfLogPath $perfLog -Key "pinned_host_input_buffers")
        Write-Host ("  kernel_time_ms={0:N6} end_to_end_ms={1:N6} graph_build_time_ms={2:N6} pinned_buffers={3}" -f $kernelMs, $endToEndMs, $graphBuildMs, $pinnedBuffers)

        Add-Result -Bucket $results -Mode $mode -Metric "kernel_time_ms" -Value $kernelMs
        Add-Result -Bucket $results -Mode $mode -Metric "end_to_end_ms" -Value $endToEndMs
        Add-Result -Bucket $results -Mode $mode -Metric "graph_build_time_ms" -Value $graphBuildMs
    }
}

$summaryPath = Join-Path $OutputDir "pinned_input_ab.csv"
$summary = New-Object System.Collections.Generic.List[string]
$summary.Add("rounds,$Rounds")
$summary.Add("metric,pageable_median_ms,pinned_median_ms,delta_pct,speedup_pageable_over_pinned")
foreach ($metricKey in $metricKeys) {
    $pageableMedian = Get-Median -Values $results["false"][$metricKey].ToArray()
    $pinnedMedian = Get-Median -Values $results["true"][$metricKey].ToArray()
    $speedup = if ($pinnedMedian -gt 0.0) { $pageableMedian / $pinnedMedian } else { [double]::NaN }
    $deltaPct = if ($pageableMedian -gt 0.0) { (($pinnedMedian - $pageableMedian) / $pageableMedian) * 100.0 } else { 0.0 }
    $summary.Add(("{0},{1:F6},{2:F6},{3:F2},{4:F4}" -f $metricKey, $pageableMedian, $pinnedMedian, $deltaPct, $speedup))
}
Set-Content -Path $summaryPath -Value $summary -Encoding ASCII

Write-Host "Done."
Write-Host "Summary: $summaryPath"
Get-Content $summaryPath
