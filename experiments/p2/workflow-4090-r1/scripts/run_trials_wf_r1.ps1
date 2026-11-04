param(
    [string[]]$Only = @()
)
# Run the workflow-4090-r1 container trials in the frozen order, serially.
$ErrorActionPreference = 'Continue'
$env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY", "User")
$env:TP_SSH_ENV_FILE = "$HOME\.tp_ssh3.env"
$root = "D:\科研\TorchPort"
$logDir = "D:\科研\Workflow-4090-r1\trial-logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$trials = @(
    @{ name = "fno-none";     bench = "FNO-Benchmark-v1";    cond = "none"; controller = "D:\科研\FNO-Benchmark-v1\controller\fno-wf-none-r1"; prompt = "prompt-fno-inference.md" },
    @{ name = "fno-full";     bench = "FNO-Benchmark-v1";    cond = "full"; controller = "D:\科研\FNO-Benchmark-v1\controller\fno-wf-full-r1"; prompt = "prompt-fno-inference.md" },
    @{ name = "simcse-none";  bench = "SimCSE-Benchmark-v1"; cond = "none"; controller = "D:\科研\SimCSE-Benchmark-v1\controller\simcse-wf-none-r1"; prompt = "prompt-simcse-inference.md" },
    @{ name = "simcse-full";  bench = "SimCSE-Benchmark-v1"; cond = "full"; controller = "D:\科研\SimCSE-Benchmark-v1\controller\simcse-wf-full-r1"; prompt = "prompt-simcse-inference.md" }
)
foreach ($trial in $trials) {
    if ($Only.Count -gt 0 -and ($Only -notcontains $trial.name)) { continue }
    $log = Join-Path $logDir ($trial.name + ".log")
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] start $($trial.name)" | Add-Content -Path $log -Encoding UTF8
    & "$root\.venv\Scripts\python.exe" "$root\experiments\harness\agents\run_container_trial.py" `
        --controller $trial.controller `
        --prompt "$root\experiments\p2\workflow-4090-r1\$($trial.prompt)" `
        --proxy http://127.0.0.1:7890 --memory 5g 2>&1 | ForEach-Object { $_ | Tee-Object -FilePath $log -Append }
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] exit=$LASTEXITCODE $($trial.name)" | Add-Content -Path $log -Encoding UTF8
}
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] TRIAL DRIVER DONE" | Add-Content -Path (Join-Path $logDir "driver.log") -Encoding UTF8
