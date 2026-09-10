# Relaunch the YOLO baseline after a host crash.
#
# On 05-09 this laptop hard-crashed mid-training and took the WSL VM with it.
# Every watchdog this project had -- watchdog.sh, yolo_supervisor.sh -- lives
# inside that VM, so nothing restarted the run and 27 hours were lost before
# anyone noticed. This registers the one trigger that survives a reboot: a
# Windows logon task, which needs no administrator rights.
#
# The task calls yolo_supervisor.sh, which is idempotent: if training is
# already running, or already finished, it does nothing and exits. Firing on
# every logon is therefore safe, and it can never stack two jobs onto the
# same 8 GB GPU.
#
#   powershell -ExecutionPolicy Bypass -File register_yolo_watchdog.ps1
#   powershell -ExecutionPolicy Bypass -File register_yolo_watchdog.ps1 -Remove
#
# Remove it once the run is scored -- it is scaffolding for this run, not a
# permanent fixture.

param([switch]$Remove)

$TaskName = 'YOLO-baseline-supervisor'
$Distro   = 'Ubuntu'
$Repo     = '/mnt/c/Users/u117134/Desktop/dev/model-v6-2'

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "removed scheduled task '$TaskName'"
    return
}

# Reuse the tmux session if the server survived; recreate it if this is a fresh
# boot. Either way the supervisor decides whether there is anything to do.
$Inner = "cd $Repo && (tmux has-session -t yolo 2>/dev/null || tmux new-session -d -s yolo -n train './yolo_supervisor.sh; exec bash')"

$Action = New-ScheduledTaskAction -Execute 'wsl.exe' -Argument "-d $Distro -- bash -lc `"$Inner`""

# A minute of slack so WSL and the GPU driver are up before the task fires.
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$Trigger.Delay = 'PT1M'

# No execution time limit: the supervisor's own launch is instant, but keeping
# the limit off means a future direct-run variant is not killed at 72 hours.
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger `
    -Settings $Settings -Force `
    -Description 'Restarts the YOLO11l-seg baseline after a host crash. Calls yolo_supervisor.sh, which no-ops if training is already running or complete.' | Out-Null

Write-Output "registered '$TaskName' (at logon, 1 min delay)"
Write-Output "  remove with: powershell -ExecutionPolicy Bypass -File register_yolo_watchdog.ps1 -Remove"
