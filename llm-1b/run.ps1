# Supervisor for the 1B NVFP4 training. Started by lm.ps1 and by the logon scheduled task.
# Restarts train.py after crashes, moves on to the next phase (exit 10), stops on PAUSE (exit 3) or done (exit 0).
# Also keeps the data download/tokenize job alive until it reports ALL DONE.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$runlog = Join-Path $logs "runner.log"
function Log($m) { Add-Content -Path $runlog -Value ("{0} {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $m) }

# single instance
$lock = Join-Path $logs "runner.pid"
if (Test-Path $lock) {
    $old = Get-Content $lock -ErrorAction SilentlyContinue
    if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) { exit 0 }
}
$PID | Out-File -Encoding ascii $lock

function Ensure-Prep {
    $plog = Join-Path $logs "prep_data.log"
    if ((Test-Path $plog) -and (Select-String -Path $plog -Pattern "ALL DONE" -Quiet)) { return }
    $ppid = Join-Path $logs "prep_data.pid"
    if (Test-Path $ppid) {
        $old = Get-Content $ppid -ErrorAction SilentlyContinue
        if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) { return }
    }
    $p = Start-Process -FilePath python -ArgumentList "-u", "prep_data.py" -WorkingDirectory $root `
        -RedirectStandardOutput $plog -RedirectStandardError (Join-Path $logs "prep_data.err") -WindowStyle Hidden -PassThru
    try { $p.PriorityClass = 'BelowNormal' } catch {}
    $p.Id | Out-File -Encoding ascii $ppid
    Log "started data prep pid $($p.Id)"
}

Log "runner start pid $PID"
$fails = 0
while ($true) {
    if (Test-Path (Join-Path $root "PAUSE")) { Log "PAUSE flag present, runner stops"; break }
    Ensure-Prep
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $out = Join-Path $logs "train_stdout_$stamp.log"
    $err = Join-Path $logs "train_stderr_$stamp.log"
    Log "launch train.py"
    $p = Start-Process -FilePath python -ArgumentList "-u", "train.py" -WorkingDirectory $root `
        -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
    $null = $p.Handle          # without touching Handle, PowerShell 5.1 loses the exit code
    $p.WaitForExit()
    $code = $p.ExitCode
    Log "train.py exited with code $code"
    if ($code -eq 0) { Log "finished"; break }
    if ($code -eq 3) { Log "paused"; break }
    if ($code -eq 10) { $fails = 0; continue }
    $fails++
    $wait = [Math]::Min(600, 30 * $fails)
    Log "crash #$fails, restarting in $wait s"
    Start-Sleep -Seconds $wait
}
Remove-Item $lock -ErrorAction SilentlyContinue
