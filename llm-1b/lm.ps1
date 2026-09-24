# Control the 1B NVFP4 training.
#   .\lm.ps1 start      start training now + auto-start at every logon (after a crash / power loss)
#   .\lm.ps1 pause      checkpoint at the end of the current step and stop (auto-start stays registered)
#   .\lm.ps1 resume     same as start
#   .\lm.ps1 status     progress, speed, ETA, last log lines
#   .\lm.ps1 log        follow the training log
#   .\lm.ps1 finish     begin the final phase now (BF16 + LR cooldown + reasoning data), then SFT
#   .\lm.ps1 stop       pause AND remove the auto-start task
param([Parameter(Position = 0)][string]$cmd = "status")
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$task = "NVFP4-1B-Train"
$pause = Join-Path $root "PAUSE"
$logs = Join-Path $root "logs"

function Start-Runner {
    Remove-Item $pause -ErrorAction SilentlyContinue
    $ps = (Get-Command powershell).Source
    $arg = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$root\run.ps1`""
    if (-not (Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue)) {
        $a = New-ScheduledTaskAction -Execute $ps -Argument $arg -WorkingDirectory $root
        $t = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
        $s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable
        Register-ScheduledTask -TaskName $task -Action $a -Trigger $t -Settings $s -Description "Auto-resume 1B NVFP4 training" | Out-Null
        Write-Host "auto-start at logon registered ($task)"
    }
    Start-Process -FilePath $ps -ArgumentList $arg -WorkingDirectory $root -WindowStyle Hidden
    Write-Host "training started (runner in background). Use .\lm.ps1 status"
}

switch ($cmd) {
    "start" { Start-Runner }
    "resume" { Start-Runner }
    "pause" {
        New-Item -ItemType File -Force $pause | Out-Null
        Write-Host "pause requested: training saves a checkpoint after the current step (up to ~1 min) and stops."
        Write-Host "resume with .\lm.ps1 resume"
    }
    "finish" {
        New-Item -ItemType File -Force (Join-Path $root "FINISH") | Out-Null
        Write-Host "final phase requested: BF16 cooldown starts at the next step."
    }
    "stop" {
        New-Item -ItemType File -Force $pause | Out-Null
        Unregister-ScheduledTask -TaskName $task -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "paused and auto-start removed."
    }
    "log" { Get-Content (Join-Path $logs "train.log") -Tail 30 -Wait }
    default {
        $sf = Join-Path $logs "status.json"
        if (Test-Path $sf) {
            $s = Get-Content $sf -Raw | ConvertFrom-Json
            $age = [int]((Get-Date).ToUniversalTime() - [DateTimeOffset]::FromUnixTimeSeconds([int64]$s.time).UtcDateTime).TotalSeconds
            "phase {0}  step {1}/{2}  tokens {3:N2}B  loss {4:N4}  {5}  {6:N0} tok/s  ETA {7:N1} h  (updated {8}s ago)" -f `
                $s.phase, $s.step, $s.total_steps, ($s.tokens / 1e9), $s.loss, $(if ($s.fp4) { "NVFP4" } else { "BF16" }), $s.tps, $s.eta_h, $age
        } else { "no status yet" }
        $running = (Test-Path (Join-Path $logs "runner.pid")) -and (Get-Process -Id (Get-Content (Join-Path $logs "runner.pid")) -ErrorAction SilentlyContinue)
        "runner: " + $(if ($running) { "running" } else { "stopped" }) + $(if (Test-Path $pause) { " (PAUSE flag set)" } else { "" })
        "auto-start task: " + $(if (Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue) { "registered" } else { "not registered" })
        $pl = Join-Path $logs "prep_data.log"
        if (Test-Path $pl) { "data: " + ((Get-Content $pl -Tail 1) -join "") }
        if (Test-Path (Join-Path $logs "train.log")) { ""; Get-Content (Join-Path $logs "train.log") -Tail 6 }
    }
}
