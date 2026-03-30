param(
    [string]$WeightsBin = "tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin",
    [string]$ParamsTemplate = "params.txt",
    [string]$OutputDir = "tests/data",
    [string]$ExePath = "build/Release/nf4_dequant.exe",
    [string]$NsysPath = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.4.2\target-windows-x64\nsys.exe",
    [int]$Rounds = 5,
    [int]$ProfileLoopIters = 1,
    $UseCudaProfilerRange = $false,
    [string]$RunTag = "input_pin"
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

$UseCudaProfilerRange = Convert-ToBoolValue -Value $UseCudaProfilerRange -Name "UseCudaProfilerRange"

function Write-ParamsWithPinnedInputFlag {
    param(
        [string]$InputPath,
        [string]$OutputPath,
        [bool]$UsePinnedHostInput,
        [int]$ProfileLoopIters
    )

    if (-not (Test-Path $InputPath)) {
        throw "params template not found: $InputPath"
    }

    $lines = Get-Content $InputPath
    $foundPinnedInput = $false
    $foundUseCudaGraph = $false
    $foundReuseBuffers = $false
    $foundKernelWarmup = $false
    $foundProfileLoop = $false
    $foundAutotune = $false
    $out = @()
    foreach ($line in $lines) {
        if ($line -match "^\s*use_pinned_host_input\s*=") {
            $out += "use_pinned_host_input = " + ($(if ($UsePinnedHostInput) { "true" } else { "false" }))
            $foundPinnedInput = $true
        } elseif ($line -match "^\s*use_cuda_graph\s*=") {
            $out += "use_cuda_graph = false"
            $foundUseCudaGraph = $true
        } elseif ($line -match "^\s*reuse_device_buffers\s*=") {
            $out += "reuse_device_buffers = true"
            $foundReuseBuffers = $true
        } elseif ($line -match "^\s*kernel_warmup_iters\s*=") {
            $out += "kernel_warmup_iters = 0"
            $foundKernelWarmup = $true
        } elseif ($line -match "^\s*profile_loop_iters\s*=") {
            $out += "profile_loop_iters = $ProfileLoopIters"
            $foundProfileLoop = $true
        } elseif ($line -match "^\s*autotune_block_dim\s*=") {
            $out += "autotune_block_dim = false"
            $foundAutotune = $true
        } else {
            $out += $line
        }
    }
    if (-not $foundPinnedInput) {
        $out += "use_pinned_host_input = " + ($(if ($UsePinnedHostInput) { "true" } else { "false" }))
    }
    if (-not $foundUseCudaGraph) {
        $out += "use_cuda_graph = false"
    }
    if (-not $foundReuseBuffers) {
        $out += "reuse_device_buffers = true"
    }
    if (-not $foundKernelWarmup) {
        $out += "kernel_warmup_iters = 0"
    }
    if (-not $foundProfileLoop) {
        $out += "profile_loop_iters = $ProfileLoopIters"
    }
    if (-not $foundAutotune) {
        $out += "autotune_block_dim = false"
    }
    Set-Content -Path $OutputPath -Value $out
}

if (-not (Test-Path $WeightsBin)) {
    throw "weights bin not found: $WeightsBin"
}
if (-not (Test-Path $ExePath)) {
    throw "executable not found: $ExePath"
}
if (-not (Test-Path $NsysPath)) {
    throw "nsys.exe not found: $NsysPath"
}
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}
if ($Rounds -le 0) {
    throw "Rounds must be > 0"
}
if ($ProfileLoopIters -le 0) {
    throw "ProfileLoopIters must be > 0."
}
if ($UseCudaProfilerRange -and $ProfileLoopIters -le 1) {
    throw "When UseCudaProfilerRange=true, ProfileLoopIters must be > 1."
}

$ProfileScript = Join-Path $PSScriptRoot "run_nsys_profile.ps1"
if (-not (Test-Path $ProfileScript)) {
    throw "run_nsys_profile.ps1 not found next to this script: $ProfileScript"
}

$baseStems = @()
$newStems = @()

if ($UseCudaProfilerRange) {
    Write-Host "[0/4][$RunTag] Running $Rounds rounds, profile_loop_iters=$ProfileLoopIters (capture last $($ProfileLoopIters - 1))."
} else {
    Write-Host "[0/4][$RunTag] Running $Rounds rounds, profile_loop_iters=$ProfileLoopIters (capture-range disabled)."
}

for ($r = 1; $r -le $Rounds; ++$r) {
    $paramsPageable = Join-Path $OutputDir ("params_nsys_{0}_pageable_r{1}.txt" -f $RunTag, $r)
    $paramsPinned = Join-Path $OutputDir ("params_nsys_{0}_pinned_r{1}.txt" -f $RunTag, $r)
    $reportPageable = Join-Path $OutputDir ("nsys_{0}_pageable_r{1}" -f $RunTag, $r)
    $reportPinned = Join-Path $OutputDir ("nsys_{0}_pinned_r{1}" -f $RunTag, $r)
    $outPageable = Join-Path $OutputDir ("out_nsys_{0}_pageable_r{1}.bin" -f $RunTag, $r)
    $outPinned = Join-Path $OutputDir ("out_nsys_{0}_pinned_r{1}.bin" -f $RunTag, $r)

    Write-Host "[1/4][$RunTag][Round $r/$Rounds] Generating params..."
    Write-ParamsWithPinnedInputFlag -InputPath $ParamsTemplate -OutputPath $paramsPageable -UsePinnedHostInput:$false -ProfileLoopIters $ProfileLoopIters
    Write-ParamsWithPinnedInputFlag -InputPath $ParamsTemplate -OutputPath $paramsPinned -UsePinnedHostInput:$true -ProfileLoopIters $ProfileLoopIters

    Write-Host "[2/4][$RunTag][Round $r/$Rounds] Nsight Systems profile (use_pinned_host_input=false)..."
    & $ProfileScript `
        -WeightsBin $WeightsBin `
        -ParamsFile $paramsPageable `
        -OutputBin $outPageable `
        -ReportStem $reportPageable `
        -ExePath $ExePath `
        -NsysPath $NsysPath `
        -WarmupFirst 0 `
        -UseCudaProfilerRange $UseCudaProfilerRange
    if ($LASTEXITCODE -ne 0) {
        throw "nsys profile failed for use_pinned_host_input=false, round=$r"
    }
    if (-not (Test-Path "$reportPageable.nsys-rep")) {
        throw "missing report: $reportPageable.nsys-rep"
    }

    Write-Host "[3/4][$RunTag][Round $r/$Rounds] Nsight Systems profile (use_pinned_host_input=true)..."
    & $ProfileScript `
        -WeightsBin $WeightsBin `
        -ParamsFile $paramsPinned `
        -OutputBin $outPinned `
        -ReportStem $reportPinned `
        -ExePath $ExePath `
        -NsysPath $NsysPath `
        -WarmupFirst 0 `
        -UseCudaProfilerRange $UseCudaProfilerRange
    if ($LASTEXITCODE -ne 0) {
        throw "nsys profile failed for use_pinned_host_input=true, round=$r"
    }
    if (-not (Test-Path "$reportPinned.nsys-rep")) {
        throw "missing report: $reportPinned.nsys-rep"
    }

    $baseStems += $reportPageable
    $newStems += $reportPinned
}

Write-Host "[4/4][$RunTag] Median comparison across $Rounds rounds..."
$compareScript = Join-Path $PSScriptRoot "compare_nsys_csv.py"
if (-not (Test-Path $compareScript)) {
    throw "compare_nsys_csv.py not found next to this script: $compareScript"
}
$compareOutput = & python $compareScript `
    --base-stems ($baseStems -join ",") `
    --new-stems ($newStems -join ",")
if ($LASTEXITCODE -ne 0) {
    throw "compare_nsys_csv failed"
}
$summaryCsv = Join-Path $OutputDir ("nsys_{0}_pinned_input_median.csv" -f $RunTag)
$compareOutput | Set-Content -Path $summaryCsv

Write-Host "Done."
Write-Host "Median summary: $summaryCsv"
foreach ($line in $compareOutput) {
    Write-Host $line
}
Write-Host "Report stems (use_pinned_host_input=false):"
foreach ($s in $baseStems) {
    Write-Host "  $s"
}
Write-Host "Report stems (use_pinned_host_input=true):"
foreach ($s in $newStems) {
    Write-Host "  $s"
}
