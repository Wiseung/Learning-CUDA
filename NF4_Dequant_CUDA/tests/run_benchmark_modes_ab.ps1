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

$modeConfigs = @(
    @{
        Name = "full"
        CopyOutput = "true"
        SaveOutput = "true"
        BenchmarkOnly = "false"
        PrimeCudaRuntime = "false"
    },
    @{
        Name = "no_save"
        CopyOutput = "true"
        SaveOutput = "false"
        BenchmarkOnly = "false"
        PrimeCudaRuntime = "false"
    },
    @{
        Name = "benchmark_only_prime"
        CopyOutput = "false"
        SaveOutput = "false"
        BenchmarkOnly = "true"
        PrimeCudaRuntime = "true"
    }
)

$metricKeys = @("wall_process_ms", "kernel_time_ms", "end_to_end_ms", "graph_build_time_ms")
$results = @{}
foreach ($cfg in $modeConfigs) {
    $results[$cfg.Name] = @{}
    foreach ($metricKey in $metricKeys) {
        $results[$cfg.Name][$metricKey] = New-Object System.Collections.Generic.List[double]
    }
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$templateLines = Get-Content $ParamsTemplate

for ($round = 1; $round -le $Rounds; ++$round) {
    $offset = ($round - 1) % $modeConfigs.Count
    for ($i = 0; $i -lt $modeConfigs.Count; ++$i) {
        $cfg = $modeConfigs[($i + $offset) % $modeConfigs.Count]
        $modeName = [string]$cfg.Name

        $paramsLines = $templateLines
        $paramsLines = Set-OrAddParamLine -Lines $paramsLines -Key "copy_output_to_host" -Value ([string]$cfg.CopyOutput)
        $paramsLines = Set-OrAddParamLine -Lines $paramsLines -Key "save_output" -Value ([string]$cfg.SaveOutput)
        $paramsLines = Set-OrAddParamLine -Lines $paramsLines -Key "benchmark_only" -Value ([string]$cfg.BenchmarkOnly)
        $paramsLines = Set-OrAddParamLine -Lines $paramsLines -Key "prime_cuda_runtime" -Value ([string]$cfg.PrimeCudaRuntime)
        $paramsLines = Set-OrAddParamLine -Lines $paramsLines -Key "use_cuda_graph" -Value "false"
        $paramsLines = Set-OrAddParamLine -Lines $paramsLines -Key "profile_loop_iters" -Value "1"
        $paramsLines = Set-OrAddParamLine -Lines $paramsLines -Key "kernel_warmup_iters" -Value "0"
        $paramsLines = Set-OrAddParamLine -Lines $paramsLines -Key "autotune_block_dim" -Value "false"

        $paramsPath = Join-Path $OutputDir ("params_{0}_r{1}.txt" -f $modeName, $round)
        $outputBin = Join-Path $OutputDir ("out_{0}_r{1}.bin" -f $modeName, $round)
        $perfLog = "$outputBin.perf.log"

        Set-Content -Path $paramsPath -Value $paramsLines -Encoding ASCII
        Write-Host ("[Round {0}/{1}] Running mode={2}" -f $round, $Rounds, $modeName)
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $ExePath $WeightsBin $paramsPath $outputBin
        $sw.Stop()
        if ($LASTEXITCODE -ne 0) {
            throw "Execution failed for mode=$modeName round=$round"
        }

        $wallMs = [double]$sw.Elapsed.TotalMilliseconds
        $kernelMs = [double](Get-PerfValue -PerfLogPath $perfLog -Key "kernel_time_ms")
        $endToEndMs = [double](Get-PerfValue -PerfLogPath $perfLog -Key "end_to_end_ms")
        $graphBuildMs = [double](Get-PerfValue -PerfLogPath $perfLog -Key "graph_build_time_ms")
        Write-Host ("  wall_process_ms={0:N6} kernel_time_ms={1:N6} end_to_end_ms={2:N6} graph_build_time_ms={3:N6}" -f $wallMs, $kernelMs, $endToEndMs, $graphBuildMs)

        Add-Result -Bucket $results -Mode $modeName -Metric "wall_process_ms" -Value $wallMs
        Add-Result -Bucket $results -Mode $modeName -Metric "kernel_time_ms" -Value $kernelMs
        Add-Result -Bucket $results -Mode $modeName -Metric "end_to_end_ms" -Value $endToEndMs
        Add-Result -Bucket $results -Mode $modeName -Metric "graph_build_time_ms" -Value $graphBuildMs
    }
}

$summaryPath = Join-Path $OutputDir "benchmark_modes_ab.csv"
$summary = New-Object System.Collections.Generic.List[string]
$summary.Add("rounds,$Rounds")
$summary.Add("metric,full_median_ms,no_save_median_ms,benchmark_only_prime_median_ms,no_save_vs_full_pct,benchmark_only_prime_vs_full_pct")
foreach ($metricKey in $metricKeys) {
    $fullMedian = Get-Median -Values $results["full"][$metricKey].ToArray()
    $noSaveMedian = Get-Median -Values $results["no_save"][$metricKey].ToArray()
    $benchMedian = Get-Median -Values $results["benchmark_only_prime"][$metricKey].ToArray()
    $noSavePct = if ($fullMedian -gt 0.0) { (($noSaveMedian - $fullMedian) / $fullMedian) * 100.0 } else { 0.0 }
    $benchPct = if ($fullMedian -gt 0.0) { (($benchMedian - $fullMedian) / $fullMedian) * 100.0 } else { 0.0 }
    $summary.Add(("{0},{1:F6},{2:F6},{3:F6},{4:F2},{5:F2}" -f $metricKey, $fullMedian, $noSaveMedian, $benchMedian, $noSavePct, $benchPct))
}
Set-Content -Path $summaryPath -Value $summary -Encoding ASCII

Write-Host "Done."
Write-Host "Summary: $summaryPath"
Get-Content $summaryPath
