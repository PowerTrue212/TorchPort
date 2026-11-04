param(
    [string[]]$Only = @()
)
# r2 retry run: the three trials that died in preflight at 11:03 (docker daemon restart +
# dead local proxy). Direct container egress to pypi works, so no --proxy is passed for any
# group (both conditions stay identical).
$ErrorActionPreference = 'Continue'
$env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY", "User")
$env:TP_SSH_ENV_FILE = "$HOME\.tp_ssh3.env"
$root = "D:\科研\TorchPort"
$logDir = "D:\科研\Workflow-4090-r1\trial-logs-r2"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$ssh = "$root\scripts\remote\ssh_exec.py"

function Get-RemoteFreeGB {
    $line = & "$root\.venv\Scripts\python.exe" $ssh run 'df -k /root/autodl-tmp | tail -1' 2>$null
    $fields = ($line -split "\s+") | Where-Object { $_ -ne '' }
    if ($fields.Count -lt 4) { return -1 }
    return [math]::Round([double]$fields[3] / 1048576, 2)
}

$trials = @(
    @{ name = "fno-none-retry"; controller = "D:\科研\FNO-Benchmark-v1\controller\fno-wf-none-r4"; prompt = "prompt-fno-inference.md"; need = 8 },
    @{ name = "lightgcn-none-retry"; controller = "D:\科研\LightGCN-Benchmark-v1\controller\lightgcn-wf-none-r2"; prompt = "prompt-lightgcn-inference.md"; need = 8 },
    @{ name = "lightgcn-full-retry"; controller = "D:\科研\LightGCN-Benchmark-v1\controller\lightgcn-wf-full-r2"; prompt = "prompt-lightgcn-inference.md"; need = 8 }
)
foreach ($trial in $trials) {
    if ($Only.Count -gt 0 -and ($Only -notcontains $trial.name)) { continue }
    $log = Join-Path $logDir ($trial.name + ".log")
    if (Test-Path (Join-Path $trial.controller "trial\result.json")) {
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] skip-already-done $($trial.name)" | Add-Content -Path $log -Encoding UTF8
        continue
    }
    $free = Get-RemoteFreeGB
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] start $($trial.name) free_gb=$free need_gb=$($trial.need)" | Add-Content -Path $log -Encoding UTF8
    if ($free -lt $trial.need) {
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] SKIP-DISK $($trial.name) free_gb=$free" | Add-Content -Path $log -Encoding UTF8
        continue
    }
    & "$root\.venv\Scripts\python.exe" "$root\experiments\harness\agents\run_container_trial.py" `
        --controller $trial.controller `
        --prompt "$root\experiments\p2\workflow-4090-r1\$($trial.prompt)" `
        --memory 5g 2>&1 |
        ForEach-Object { $_ | Tee-Object -FilePath $log -Append }
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] exit=$LASTEXITCODE $($trial.name)" | Add-Content -Path $log -Encoding UTF8
}
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] RETRY DRIVER DONE" | Add-Content -Path (Join-Path $logDir "driver-retry.log") -Encoding UTF8
