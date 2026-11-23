param([Parameter(Mandatory)][string]$BenchmarkDll,[string]$Output='artifacts/gpu-concurrency-followup')
$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Force $Output | Out-Null
$env:SIMD_OCR_BENCH_WARMUP_SECONDS='8'
$powerBefore=(powercfg /getactivescheme | Out-String).Trim()
foreach($config in @(@('A',1),@('B',1),@('C',1),@('C',6),@('B',6),@('A',6))) {
    $mode=$config[0]; $degree=$config[1]
    $env:SIMD_OCR_EXPERIMENT_COMPACT_HEAD=if($mode -eq 'A'){'0'}else{'1'}
    $env:SIMD_OCR_EXPERIMENT_SINGLE_WAVE=if($mode -eq 'C'){'1'}else{'0'}
    $name="confirm-$mode-c$degree"
    dotnet $BenchmarkDll F:/Docs/Quicker/_packages/quicker.ocr.v6small/0.2.0 D:/Work/qk-ocr-lite/fixtures/images (Join-Path $Output "$name.json") 30 $degree 0 0 --decode *> (Join-Path $Output "$name.log")
    if($LASTEXITCODE -ne 0){throw "Benchmark failed: $name"}
    $r=Get-Content (Join-Path $Output "$name.json") -Raw | ConvertFrom-Json
    if($r.Mismatches -ne 0){throw "Unstable result: $name"}
    Write-Output "$name $($r.ImagesPerSecond) img/s $($r.MeanMs) ms"
    if((powercfg /getactivescheme | Out-String).Trim() -ne $powerBefore){throw 'Power scheme changed.'}
}
Remove-Item Env:SIMD_OCR_EXPERIMENT_COMPACT_HEAD,Env:SIMD_OCR_EXPERIMENT_SINGLE_WAVE,Env:SIMD_OCR_BENCH_WARMUP_SECONDS
