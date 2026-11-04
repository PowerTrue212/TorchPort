param(
    [string[]]$Only = @()
)
# r2 retry run v2: WSL VM was idle-shutting-down every ~60 s, which killed dockerd and the
# previous driver (killed with the DSH job runner). A `wsl -- sleep infinity` keepalive is
# kept alive and this script is launched detached so neither event stops the trials.
$ErrorActionPreference = 'Continue'
$env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY", "User")
$env:TP_SSH_ENV_FILE = "$HOME\.tp_ssh3.env"
$root = "D:\科研\TorchPort"
$logDir = "D:\科研\Workflow-4090-r1\trial-logs-r2"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$ssh = "$root\scripts\remote\ssh_exec.py"
$driverLog = Join-Path $logDir "driver-retry-v2.log"

function Log($text) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $text" | Add-Content -Path $driverLog -Encoding UTF8
}

function Get-RemoteFreeGB {
    $line = & "$root\.venv\Scripts\python.exe" $ssh run 'df -k /root/autodl-tmp | tail -1' 2>$null
    $fields = ($line -split "\s+") | Where-Object { $_ -ne '' }
    if ($fields.Count -lt 4) { return -1 }
    return [math]::Round([double]$fields[3] / 1048576, 2)
}

# keep the WSL VM (and dockerd) alive for the whole run
$keepalive = Get-Process -Name wsl -ErrorAction SilentlyContinue
Log "wsl processes before start: $(($keepalive | Measure-Object).Count)"

$trials = @(
    @{ name = "fno-none-retry2"; controller = "D:\科研\FNO-Benchmark-v1\controller\fno-wf-none-r5"; prompt = "prompt-fno-inference.md"; need = 8 },
    @{ name = "lightgcn-none-retry"; controller = "D:\科研\LightGCN-Benchmark-v1\controller\lightgcn-wf-none-r2"; prompt = "prompt-lightgcn-inference.md"; need = 8 },
    @{ name = "lightgcn-full-retry"; controller = "D:\科研\LightGCN-Benchmark-v1\controller\lightgcn-wf-full-r2"; prompt = "prompt-lightgcn-inference.md"; need = 8 }
)
foreach ($trial in $trials) {
    if ($Only.Count -gt 0 -and ($Only -notcontains $trial.name)) { continue }
    $log = Join-Path $logDir ($trial.name + ".log")
    if (Test-Path (Join-Path $trial.controller "trial\result.json")) {
        Log "skip-already-done $($trial.name)"
        continue
    }
    # restart docker if the VM recycled it
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
Log "RETRY DRIVER V2 DONE"
