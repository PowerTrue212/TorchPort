param(
    [string[]]$Only = @()
)
# Workflow r3 driver: RAFT none -> RAFT full -> FNO none, serially, detached from the DSH job
# runner. Keeps the WSL VM alive (r2 lost dockerd to idle VM recycling) and checks remote disk
# before each trial. A trial that already has trial\result.json is never rerun.
$ErrorActionPreference = 'Continue'
$env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY", "User")
$env:TP_SSH_ENV_FILE = "$HOME\.tp_ssh3.env"
$root = "D:\科研\TorchPort"
$logDir = "D:\科研\Workflow-4090-r1\trial-logs-r3"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$ssh = "$root\scripts\remote\ssh_exec.py"
$driverLog = Join-Path $logDir "driver-r3.log"

function Log($text) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $text" | Add-Content -Path $driverLog -Encoding UTF8
}

function Get-RemoteFreeGB {
    $line = & "$root\.venv\Scripts\python.exe" $ssh run 'df -k /root/autodl-tmp | tail -1' 2>$null
    $fields = ($line -split "\s+") | Where-Object { $_ -ne '' }
    if ($fields.Count -lt 4) { return -1 }
    return [math]::Round([double]$fields[3] / 1048576, 2)
}

$keepalive = Start-Process -FilePath wsl.exe -ArgumentList '-d', 'Ubuntu', '--', 'sleep', 'infinity' -WindowStyle Hidden -PassThru
Log "keepalive pid=$($keepalive.Id)"

# need = remote free GB required before the trial starts (r2 RAFT none wrote ~7.5 GB)
$trials = @(
    @{ name = "raft-none"; controller = "D:\科研\RAFT-Benchmark-v1\controller\raft-wf-none-r3"; prompt = "prompt-raft.md"; need = 9 },
    @{ name = "raft-full"; controller = "D:\科研\RAFT-Benchmark-v1\controller\raft-wf-full-r3"; prompt = "prompt-raft.md"; need = 9 },
    @{ name = "fno-none"; controller = "D:\科研\FNO-Benchmark-v1\controller\fno-wf-none-r6"; prompt = "prompt-fno-inference.md"; need = 3 }
)
foreach ($trial in $trials) {
    if ($Only.Count -gt 0 -and ($Only -notcontains $trial.name)) { continue }
    $log = Join-Path $logDir ($trial.name + ".log")
    if (Test-Path (Join-Path $trial.controller "trial\result.json")) {
        Log "skip-already-done $($trial.name)"
        continue
    }
    if (Test-Path (Join-Path $trial.controller "canaries.json")) {
        Log "skip-started-before $($trial.name) (canaries.json exists; never resume a started trial)"
        continue
    }
    & wsl.exe -d Ubuntu -u root -- service docker start 2>&1 | Out-Null
    Start-Sleep -Seconds 5
    $free = Get-RemoteFreeGB
    Log "start $($trial.name) free_gb=$free need_gb=$($trial.need)"
    if ($free -lt $trial.need) {
        Log "SKIP-DISK $($trial.name) free_gb=$free"
        continue
    }
    & "$root\.venv\Scripts\python.exe" "$root\experiments\harness\agents\run_container_trial.py" `
        --controller $trial.controller `
        --prompt "$root\experiments\p2\workflow-4090-r1\$($trial.prompt)" `
        --memory 5g 2>&1 |
        ForEach-Object { $_ | Tee-Object -FilePath $log -Append }
    Log "exit=$LASTEXITCODE $($trial.name)"
}
Log "R3 DRIVER DONE"
