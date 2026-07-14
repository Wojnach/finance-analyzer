# LLM backtest runner (herc2). Lifts the local-LLM gate flag for the run,
# always restores it, logs to data\llm_backtest_run.log.
# Invoked by the Deck orchestrator via a one-shot scheduled task:
#   scripts\deck\run-llm-backtest-on-herc.sh
param(
    [string]$Models = "ministral3,qwen3,phi4_mini,fin_r1",
    [string]$Start = "2026-02-01",
    [string]$End = "2026-07-11",
    [int]$StepHours = 8,
    [string]$Out = "data\llm_backtest_results.jsonl",
    [string]$Tickers = "BTC-USD,ETH-USD",
    [string]$Interval = "1h",
    [switch]$KeepRaw,
    [switch]$Headlines,
    [switch]$FromArgsFile
)

# schtasks /tr silently truncates ~261 chars — long invocations must go
# through the args file (written by the Deck orchestrator) and a short
# fixed task command with just -FromArgsFile.
if ($FromArgsFile) {
    $af = "Q:\finance-analyzer\data\llm_backtest_args.json"
    if (Test-Path $af) {
        $a = Get-Content $af -Raw | ConvertFrom-Json
        $Models = $a.models; $Start = $a.start; $End = $a.end
        $StepHours = [int]$a.step_hours; $Out = $a.out
        $Tickers = $a.tickers; $Interval = $a.interval
        if ($a.keep_raw) { $KeepRaw = $true }
        if ($a.headlines) { $Headlines = $true }
    } else {
        Write-Host "args file missing: $af"; exit 1
    }
}

$repo = "Q:\finance-analyzer"
$gate = Join-Path $repo "data\local_llm.disabled"
$gateLifted = "$gate.backtest-lifted"
$log = Join-Path $repo "data\llm_backtest_run.log"

Start-Transcript -Path $log -Append
try {
    if (Test-Path $gate) {
        Rename-Item $gate $gateLifted
        Write-Host "gate lifted"
    }
    Set-Location $repo
    $extra = @()
    if ($KeepRaw) { $extra += "--keep-raw" }
    if ($Headlines) { $extra += "--headlines" }
    & .venv\Scripts\python.exe -u scripts\llm_backtest.py `
        --models $Models --start $Start --end $End --step-hours $StepHours `
        --tickers $Tickers --interval $Interval --out $Out @extra
    Write-Host "exit code: $LASTEXITCODE"
} finally {
    if (Test-Path $gateLifted) {
        Rename-Item $gateLifted $gate
        Write-Host "gate restored"
    }
    Stop-Transcript
}
