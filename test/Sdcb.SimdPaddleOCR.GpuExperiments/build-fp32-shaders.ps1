param([string]$Compiler)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path "$PSScriptRoot/../..").Path
if (!$Compiler) { $Compiler = "$repoRoot/artifacts/gpu-experiments/glslang/bin/glslang.exe" }
$shaderRoot = "$repoRoot/src/Sdcb.SimdPaddleOCR/Experimental/Shaders"
foreach ($halfMode in @($false, $true)) {
    $storageArgs = @(); $suffix = ''
    if ($halfMode) { $storageArgs = @('-DF16=1'); $suffix = '_h' }
    & $Compiler -V @storageArgs '-DLOGITS_FP32=1' "$shaderRoot/ctc_projection.comp" -o "$shaderRoot/ctc_projection32$suffix.spv"
    if ($LASTEXITCODE) { throw 'FP32 projection shader compilation failed' }
    & $Compiler -V @storageArgs "$shaderRoot/ctc_projection.comp" -o "$shaderRoot/ctc_projection16$suffix.spv"
    if ($LASTEXITCODE) { throw 'FP16 control shader compilation failed' }
    & $Compiler -V @storageArgs "$shaderRoot/ctc_reduce32.comp" -o "$shaderRoot/ctc_reduce32$suffix.spv"
    if ($LASTEXITCODE) { throw 'FP32 reduction shader compilation failed' }
}
