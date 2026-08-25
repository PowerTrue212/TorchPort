[CmdletBinding()]
param(
    [switch]$AuditOnly,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$DshArgs
)

$launcher = Join-Path $PSScriptRoot "scripts\launch_dsh_pure.ps1"
if ($null -ne $DshArgs -and $DshArgs.Count -gt 0) {
    & $launcher -AuditOnly:$AuditOnly -DshArgs $DshArgs
}
else {
    & $launcher -AuditOnly:$AuditOnly
}
exit $LASTEXITCODE
