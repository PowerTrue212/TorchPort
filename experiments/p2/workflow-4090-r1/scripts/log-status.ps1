param(
    [Parameter(Mandatory=$true)][string]$Step,
    [Parameter(Mandatory=$true)][ValidateSet('PASS','FAIL','SKIP','INFO')][string]$State,
    [Parameter(Mandatory=$true)][string]$OutputFile,
    [string]$MaxLines = '40',
    [string]$StatusFile = 'D:\科研\Workflow-4090-r1\status.md'
)
# Append one step result block to the status log in the README format.
# The output file's first line must be "命令：<command line>"; the rest is raw output.
$ErrorActionPreference = 'Stop'
$dir = Split-Path -Parent $StatusFile
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
if (-not (Test-Path $StatusFile)) {
    $hdr = "# Workflow 4090 r1 status`n`nStarted: " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "`n"
    Set-Content -Path $StatusFile -Value $hdr -Encoding utf8
}
$all = @()
if (Test-Path $OutputFile) { $all = @(Get-Content -Path $OutputFile -Encoding utf8) }
$cmd = '(未记录)'
if ($all.Count -gt 0 -and $all[0].StartsWith('命令：')) {
    $cmd = $all[0].Substring(3)
    $all = @($all | Select-Object -Skip 1)
}
$block = New-Object System.Collections.Generic.List[string]
$block.Add('')
$block.Add('## ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ' + $Step + ' ' + $State)
$block.Add('命令：' + $cmd)
$n = [int]$MaxLines
if ($all.Count -gt $n) {
    $block.Add('关键输出（原样，最多 ' + $n + ' 行，截断为' + '尾部）：')
    $block.AddRange([string[]]($all[($all.Count - $n)..($all.Count - 1)]))
} else {
    $block.Add('关键输出（原样，最多 ' + $n + ' 行）：')
    $block.AddRange([string[]]$all)
}
Add-Content -Path $StatusFile -Value (($block -join "`n") + "`n") -Encoding utf8
Write-Output ("logged: " + $Step + " " + $State)
