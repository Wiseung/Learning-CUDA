param(
    [string]$WeightsBin = "tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin",
    [string]$ParamsFile = "tests/data/params_bd256.txt",
    [string]$OutputBin = "tests/data/out_nsys.bin",
    [string]$ReportStem = "tests/data/nsys_run",
    [string]$ExePath = "build/Release/nf4_dequant.exe",
    [string]$NsysPath = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.4.2\target-windows-x64\nsys.exe",
    $WarmupFirst = $true,
    $UseCudaProfilerRange = $false
)

$ErrorActionPreference = "Stop"

function Convert-ToBoolValue {
    param(
        $Value,
        [string]$Name
    )

    if ($Value -is [bool]) {
        return $Value
    }
    if ($Value -is [int] -or $Value -is [long]) {
        if ($Value -eq 0) { return $false }
        if ($Value -eq 1) { return $true }
    }

    $text = "$Value".Trim().ToLowerInvariant()
    switch ($text) {
        "true" { return $true }
        "false" { return $false }
        "1" { return $true }
        "0" { return $false }
        '$true' { return $true }
        '$false' { return $false }
    }
    throw ("Invalid boolean value for {0}: {1}" -f $Name, $Value)
}

$WarmupFirst = Convert-ToBoolValue -Value $WarmupFirst -Name "WarmupFirst"
$UseCudaProfilerRange = Convert-ToBoolValue -Value $UseCudaProfilerRange -Name "UseCudaProfilerRange"

if (-not (Test-Path $NsysPath)) {
    throw "nsys.exe not found: $NsysPath"
}
if (-not (Test-Path $ExePath)) {
    throw "Executable not found: $ExePath"
}
if (-not (Test-Path $WeightsBin)) {
    throw "weights bin not found: $WeightsBin"
}
if (-not (Test-Path $ParamsFile)) {
    throw "params file not found: $ParamsFile"
}

if ($WarmupFirst) {
    Write-Host "[0/2] Warmup run (not profiled)..."
    $warmupOut = "${OutputBin}.warmup.bin"
    & $ExePath $WeightsBin $ParamsFile $warmupOut | Out-Null
}

Write-Host "[1/2] Profiling with Nsight Systems..."
$profileArgs = @(
    "profile",
    "--force-overwrite=true",
    "--trace=cuda",
    "--sample=none",
    "--cpuctxsw=none",
    "--stats=true",
    "-o", $ReportStem
)
if ($UseCudaProfilerRange) {
    $profileArgs += "--capture-range=cudaProfilerApi"
    $profileArgs += "--capture-range-end=stop"
}
$profileArgs += @($ExePath, $WeightsBin, $ParamsFile, $OutputBin)
& $NsysPath @profileArgs

$repFile = "$ReportStem.nsys-rep"
if (-not (Test-Path $repFile)) {
    throw "Expected report not found: $repFile"
}

Write-Host "[2/2] Exporting stats reports..."
& $NsysPath stats `
    --force-export=true `
    --force-overwrite=true `
    --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum `
    --format csv,csv,csv,csv `
    --output . `
    $repFile

Write-Host "Done."
Write-Host "Report file: $repFile"
Write-Host "Stats files:"
Write-Host "  ${ReportStem}_cuda_api_sum.csv"
Write-Host "  ${ReportStem}_cuda_gpu_kern_sum.csv"
Write-Host "  ${ReportStem}_cuda_gpu_mem_time_sum.csv"
Write-Host "  ${ReportStem}_cuda_gpu_mem_size_sum.csv"
