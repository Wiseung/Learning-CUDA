param(
    [string]$Dtype = "fp16",
    [int]$HeadDim = 64,
    [int]$TotalRows = 65536,
    [int]$Iterations = 3,
    [int]$Warmup = 1,
    [string]$Mode = "hadamard",
    [string]$Output = "hadamard_timeline"
)

nsys profile --stats=true -o $Output `
    .\build\Release\hadamard_bench.exe `
    --dtype $Dtype --head-dim $HeadDim --total-rows $TotalRows `
    --iterations $Iterations --warmup $Warmup --mode $Mode
