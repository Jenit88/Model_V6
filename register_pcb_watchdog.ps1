# Relaunch the model_v6_2 training run after a host crash.
#
# On 09-09 this laptop hard-crashed at epoch 32/120 of the Tier-1 run and took
# the WSL VM with it. watchdog.sh polls from inside that VM, so it died with
# the job it was supposed to restart -- the same way 27 hours were lost on
# 05-09. YOLO already had this task by then and stood down correctly; the
# training run had no equivalent, so nothing brought it back.
#
# This registers the one trigger that survives a reboot: a Windows logon task,
# which needs no administrator rights.
#
# The task calls pcb_supervisor.sh, which is idempotent: if training is already
# running, already finished, or deliberately stopped, it does nothing. Firing
# on every logon is therefore safe, and it can never stack two jobs onto the
# same 8 GB GPU.
#
#   powershell -ExecutionPolicy Bypass -File register_pcb_watchdog.ps1
#   powershell -ExecutionPolicy Bypass -File register_pcb_watchdog.ps1 -Remove
#
# Remove it once the run is scored -- it is scaffolding for this run, not a
# permanent fixture.

param([switch]$Remove)

$TaskName = 'PCB-v62-training-supervisor'
$Distro   = 'Ubuntu'
$Repo     = '/mnt/c/Users/u117134/Desktop/dev/model-v6-2'

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "removed scheduled task '$TaskName'"
    return
}

# The supervisor blocks for 240 s while it confirms training came up, so it must
# run detached -- a logon task that sits for four minutes delays the logon.
#
# It is detached into tmux, NOT with `nohup ... &`. The obvious form,
#   wsl.exe -- bash -lc "nohup ./pcb_supervisor.sh >> log 2>&1 &"
# returns instantly with exit code 0 and the supervisor never runs: wsl.exe
# tears the session down as soon as that bash exits, taking the backgrounded
# child with it. Task Scheduler records LastTaskResult 0 and the log stays
# empty, so the whole mechanism looks healthy while doing nothing -- the same
# shape of silent failure that cost 27 hours on 05-09. Measured on 09-09.
#
# The tmux server is a daemon and outlives wsl.exe, which is why the YOLO task
# uses this form too. Killing the `pcbsup` session first means every logon
# really re-runs the supervisor; that is safe because the supervisor no-ops
# when training is already alive, and `pcbsup` never holds the training job --
# that lives in the `pcb` session and is untouched here.
$Inner = "cd $Repo && (tmux kill-session -t pcbsup 2>/dev/null; tmux new-session -d -s pcbsup -n sup './pcb_supervisor.sh; exec bash')"

$Action = New-ScheduledTaskAction -Execute 'wsl.exe' -Argument "-d $Distro -- bash -lc `"$Inner`""

# Two minutes of slack: WSL, the GPU driver, and the /mnt/c mount all need to
# be up before the supervisor can read training_log.csv or see the GPU.
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$Trigger.Delay = 'PT2M'

$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger `
    -Settings $Settings -Force `
    -Description 'Resumes the model_v6_2 training run after a host crash. Calls pcb_supervisor.sh, which no-ops if training is already running, complete, or deliberately stopped.' | Out-Null

Write-Output "registered '$TaskName' (at logon, 2 min delay)"
Write-Output "  remove with: powershell -ExecutionPolicy Bypass -File register_pcb_watchdog.ps1 -Remove"
