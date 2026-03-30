param(
    [string]$WeightsBin = "tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin",
    [string]$ParamsTemplate = "tests/data/params_nsys_pinned.txt",
    [string]$OutputDir = "tests/data",
    [string]$ExePath = "build/Release/nf4_dequant.exe",
    [string]$NsysPath = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.4.2\target-windows-x64\nsys.exe",
    [int]$Rounds = 5,
    [int]$ProfileLoopIters = 6,
    $UseCudaProfilerRange = $true,
    [string]$RunTag = "variant"
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

function Write-ParamsWithKernelVariant {
    param(
        [string]$InputPath,
        [string]$OutputPath,
        [string]$KernelVariant,
        [int]$ProfileLoopIters
    )

    if (-not (Test-Path $InputPath)) {
        throw "params template not found: $InputPath"
    }

    $lines = Get-Content $InputPath
    $foundVariant = $false
    $foundKernelWarmup = $false
    $foundProfileLoop = $false
    $foundAutotune = $false
    $out = @()
    foreach ($line in $lines) {
        if ($line -match "^\s*kernel_variant\s*=") {
            $out += "kernel_variant = `"$KernelVariant`""
            $foundVariant = $true
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
    if (-not $foundVariant) {
        $out += "kernel_variant = `"$KernelVariant`""
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

$baseStems = @()
$newStems = @()

if ($UseCudaProfilerRange) {
    Write-Host "[0/4][$RunTag] Running $Rounds rounds, profile_loop_iters=$ProfileLoopIters (capture last $($ProfileLoopIters - 1))."
} else {
    Write-Host "[0/4][$RunTag] Running $Rounds rounds, profile_loop_iters=$ProfileLoopIters (capture-range disabled)."
}

for ($r = 1; $r -le $Rounds; ++$r) {
    $paramsGeneric = Join-Path $OutputDir ("params_nsys_{0}_generic_r{1}.txt" -f $RunTag, $r)
    $paramsSpecialized = Join-Path $OutputDir ("params_nsys_{0}_specialized_r{1}.txt" -f $RunTag, $r)
    $reportGeneric = Join-Path $OutputDir ("nsys_{0}_generic_r{1}" -f $RunTag, $r)
    $reportSpecialized = Join-Path $OutputDir ("nsys_{0}_specialized_r{1}" -f $RunTag, $r)
    $outGeneric = Join-Path $OutputDir ("out_nsys_{0}_generic_r{1}.bin" -f $RunTag, $r)
    $outSpecialized = Join-Path $OutputDir ("out_nsys_{0}_specialized_r{1}.bin" -f $RunTag, $r)

    Write-Host "[1/4][$RunTag][Round $r/$Rounds] Generating params..."
    Write-ParamsWithKernelVariant -InputPath $ParamsTemplate -OutputPath $paramsGeneric -KernelVariant "generic" -ProfileLoopIters $ProfileLoopIters
    Write-ParamsWithKernelVariant -InputPath $ParamsTemplate -OutputPath $paramsSpecialized -KernelVariant "specialized" -ProfileLoopIters $ProfileLoopIters

    Write-Host "[2/4][$RunTag][Round $r/$Rounds] Nsight Systems profile (kernel_variant=generic)..."
    & tests/run_nsys_profile.ps1 `
        -WeightsBin $WeightsBin `
        -ParamsFile $paramsGeneric `
        -OutputBin $outGeneric `
        -ReportStem $reportGeneric `
        -ExePath $ExePath `
        -NsysPath $NsysPath `
        -WarmupFirst 0 `
        -UseCudaProfilerRange $UseCudaProfilerRange
    if ($LASTEXITCODE -ne 0) {
        throw "nsys profile failed for kernel_variant=generic, round=$r"
    }
    if (-not (Test-Path "$reportGeneric.nsys-rep")) {
        throw "missing report: $reportGeneric.nsys-rep"
    }

    Write-Host "[3/4][$RunTag][Round $r/$Rounds] Nsight Systems profile (kernel_variant=specialized)..."
    & tests/run_nsys_profile.ps1 `
        -WeightsBin $WeightsBin `
        -ParamsFile $paramsSpecialized `
        -OutputBin $outSpecialized `
        -ReportStem $reportSpecialized `
        -ExePath $ExePath `
        -NsysPath $NsysPath `
        -WarmupFirst 0 `
        -UseCudaProfilerRange $UseCudaProfilerRange
    if ($LASTEXITCODE -ne 0) {
        throw "nsys profile failed for kernel_variant=specialized, round=$r"
    }
    if (-not (Test-Path "$reportSpecialized.nsys-rep")) {
        throw "missing report: $reportSpecialized.nsys-rep"
    }

    $baseStems += $reportGeneric
    $newStems += $reportSpecialized
}

Write-Host "[4/4][$RunTag] Median comparison across $Rounds rounds..."
$compareOutput = & python tests/compare_nsys_csv.py `
    --base-stems ($baseStems -join ",") `
    --new-stems ($newStems -join ",")
if ($LASTEXITCODE -ne 0) {
    throw "compare_nsys_csv failed"
}
$summaryCsv = Join-Path $OutputDir ("nsys_{0}_kernel_variant_median.csv" -f $RunTag)
$compareOutput | Set-Content -Path $summaryCsv

Write-Host "Done."
Write-Host "Median summary: $summaryCsv"
foreach ($line in $compareOutput) {
    Write-Host $line
}
Write-Host "Report stems (kernel_variant=generic):"
foreach ($s in $baseStems) {
    Write-Host "  $s"
}
Write-Host "Report stems (kernel_variant=specialized):"
foreach ($s in $newStems) {
    Write-Host "  $s"
}
