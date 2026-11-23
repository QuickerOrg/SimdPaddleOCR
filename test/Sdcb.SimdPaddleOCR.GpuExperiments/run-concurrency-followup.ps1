param(
    [Parameter(Mandatory)][string]$BenchmarkDll,
    [string]$Output='artifacts/gpu-concurrency-followup',
    [int]$Seconds=12,
    [int]$WarmupSeconds=8,
    [int]$Rounds=2
)
$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Force $Output | Out-Null
$planBefore=(powercfg /getactivescheme | Out-String).Trim()
$planBefore | Set-Content (Join-Path $Output 'power-plan-before.txt') -Encoding utf8
$env:SIMD_OCR_BENCH_WARMUP_SECONDS=[string]$WarmupSeconds
$models='F:/Docs/Quicker/_packages/quicker.ocr.v6small/0.2.0'
$images='D:/Work/qk-ocr-lite/fixtures/images'
$records=@()
foreach($round in 1..$Rounds) {
    $degrees=if($round%2 -eq 1){@(1,2,4,6,8)}else{@(8,6,4,2,1)}
    $modes=if($round%2 -eq 1){@('A','B','C')}else{@('C','B','A')}
    foreach($degree in $degrees) {
        foreach($mode in $modes) {
            $env:SIMD_OCR_EXPERIMENT_COMPACT_HEAD=if($mode -eq 'A'){'0'}else{'1'}
            $env:SIMD_OCR_EXPERIMENT_SINGLE_WAVE=if($mode -eq 'C'){'1'}else{'0'}
            $name="$mode-c$degree-r$round"
            Write-Output "START $name $(Get-Date -Format o)"
            dotnet $BenchmarkDll $models $images (Join-Path $Output "$name.json") $Seconds $degree 0 0 --decode *> (Join-Path $Output "$name.log")
            if($LASTEXITCODE -ne 0){throw "Benchmark failed: $name"}
            $r=Get-Content (Join-Path $Output "$name.json") -Raw | ConvertFrom-Json
            if($r.Mismatches -ne 0){throw "Unstable result: $name ($($r.Mismatches))"}
            $records+=[pscustomobject]@{Name=$name;Mode=$mode;Concurrency=$degree;Round=$round;ImagesPerSecond=$r.ImagesPerSecond;MeanMs=$r.MeanMs;CpuCores=$r.CpuCores}
            $records | ConvertTo-Json | Set-Content (Join-Path $Output 'progress.json') -Encoding utf8
            Write-Output "DONE $name $([Math]::Round($r.ImagesPerSecond,2)) img/s $([Math]::Round($r.MeanMs,2)) ms cpu=$([Math]::Round($r.CpuCores,2))"
            $planNow=(powercfg /getactivescheme | Out-String).Trim()
            if($planNow -ne $planBefore){throw 'Power plan changed during the scan.'}
        }
    }
}
$planNow | Set-Content (Join-Path $Output 'power-plan-after.txt') -Encoding utf8
Remove-Item Env:SIMD_OCR_EXPERIMENT_COMPACT_HEAD,Env:SIMD_OCR_EXPERIMENT_SINGLE_WAVE,Env:SIMD_OCR_BENCH_WARMUP_SECONDS
