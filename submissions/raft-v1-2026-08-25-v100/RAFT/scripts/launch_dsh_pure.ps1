[CmdletBinding()]
param(
    [switch]$AuditOnly,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$DshArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$experimentRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$workspace = Join-Path $experimentRoot "workspaces\dsh\repo"
$overlay = Join-Path $experimentRoot "isolation\dsh-pure.patch.yml"
$auditDirectory = Join-Path $experimentRoot "results\dsh\isolation"
$auditPath = Join-Path $auditDirectory "latest.json"
$dshProfile = "web"
$forbiddenEnvironment = @(
    "TORCHPORT_ROOT",
    "TORCHPORT_PYTHON",
    "TORCHPORT_PYTHONPATH",
    "TORCHPORT_TRACE_DIR"
)

foreach ($requiredPath in @($workspace, $overlay)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required isolation path does not exist: $requiredPath"
    }
}

# Use the cmd shim so the DSH launcher and its environment always run in a
# child process.  Calling the PowerShell npm shim can leak process-local state
# back into this preflight process.
$dshCommand = Get-Command dsh.cmd -ErrorAction Stop
$savedEnvironment = @{}
$originalLocation = (Get-Location).Path
$audit = [ordered]@{
    schema_version = 1
    generated_at_utc = [DateTime]::UtcNow.ToString("o")
    method = "dsh"
    profile = $dshProfile
    workspace = $workspace
    overlay = $overlay
    overlay_sha256 = (Get-FileHash -LiteralPath $overlay -Algorithm SHA256).Hash.ToLowerInvariant()
    dsh_command = $dshCommand.Source
    installed_torchport_plugin = $null
    active_torchport_plugin = $null
    overlay_disabled = $null
    forbidden_environment_removed = @()
    passed = $false
}

try {
    foreach ($name in $forbiddenEnvironment) {
        $value = [Environment]::GetEnvironmentVariable($name, "Process")
        $savedEnvironment[$name] = $value
        if (-not [string]::IsNullOrEmpty($value)) {
            $audit.forbidden_environment_removed += $name
        }
        [Environment]::SetEnvironmentVariable($name, $null, "Process")
    }

    $version = (& $dshCommand.Source --version 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to query DSH version: $version"
    }
    $audit.dsh_version = $version

    $inventory = (& $dshCommand.Source plugin --profile $dshProfile list --depth 0 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect DSH plugin inventory: $inventory"
    }
    $audit.installed_torchport_plugin = [bool]($inventory -match "@torchport/dsh-plugin@")

    # DSH uses the same composition code for --dump-config and startup.  Audit
    # that exact composed tree before allowing an interactive session.
    $composed = (& $dshCommand.Source --profile $dshProfile --patch $overlay --dump-config 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to compose isolated DSH configuration: $composed"
    }

    $torchportBlock = [regex]::Match(
        $composed,
        "(?ms)^- id: torchport\r?\n(?<body>.*?)(?=^- id: |\z)"
    )
    if (-not $torchportBlock.Success) {
        throw "Isolation overlay did not match the installed TorchPort entry."
    }

    $body = $torchportBlock.Groups["body"].Value
    $audit.overlay_disabled = [bool]($body -match "(?m)^  disabled: true\s*$")
    $audit.active_torchport_plugin = -not $audit.overlay_disabled
    if ($audit.active_torchport_plugin) {
        throw "TorchPort remains active in the composed DSH configuration."
    }

    foreach ($name in $forbiddenEnvironment) {
        $survivingValue = [Environment]::GetEnvironmentVariable($name, "Process")
        if (-not [string]::IsNullOrEmpty($survivingValue)) {
            throw "Forbidden environment variable survived sanitization: $name"
        }
    }

    $audit.passed = $true
    New-Item -ItemType Directory -Path $auditDirectory -Force | Out-Null
    $audit | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $auditPath -Encoding UTF8
    Write-Host "Pure DSH isolation audit passed."
    Write-Host "  TorchPort installed: $($audit.installed_torchport_plugin)"
    Write-Host "  TorchPort active:    $($audit.active_torchport_plugin)"
    Write-Host "  Audit:               $auditPath"

    if ($AuditOnly) {
        return
    }

    Set-Location -LiteralPath $workspace
    & $dshCommand.Source --profile $dshProfile --patch $overlay @DshArgs
    if ($LASTEXITCODE -ne 0) {
        throw "The isolated DSH process exited with code $LASTEXITCODE."
    }
}
catch {
    $audit.error = $_.Exception.Message
    New-Item -ItemType Directory -Path $auditDirectory -Force | Out-Null
    $audit | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $auditPath -Encoding UTF8
    throw
}
finally {
    Set-Location -LiteralPath $originalLocation
    foreach ($name in $forbiddenEnvironment) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], "Process")
    }
}
