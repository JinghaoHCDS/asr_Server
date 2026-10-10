# Start the already-installed user service. No dependency or model installation occurs here.
[CmdletBinding()]
param(
    [string]$Distro = $env:ASR_WSL_DISTRO,
    [string]$LinuxUser = $env:ASR_WSL_USER,
    [string]$ProjectDir = $env:ASR_WSL_PROJECT_DIR,
    [ValidatePattern('^[A-Za-z0-9_.@-]+\.service$')]
    [string]$ServiceName = 'asr-server.service',
    [switch]$NoKeepAlive
)

$ErrorActionPreference = 'Stop'

function ConvertTo-BashLiteral([string]$Value) {
    return "'" + $Value.Replace("'", "'\''") + "'"
}

if ($ProjectDir) {
    if (-not $ProjectDir.StartsWith('/') -or $ProjectDir.StartsWith('/mnt/') -or $ProjectDir -match '[\r\n]') {
        throw 'ProjectDir must be an absolute path in the WSL Linux filesystem.'
    }
    $ProjectExpression = ConvertTo-BashLiteral $ProjectDir
} else {
    $ProjectExpression = '"$HOME/services/asr-server"'
}

$ServiceLiteral = ConvertTo-BashLiteral $ServiceName
$Command = 'set -euo pipefail; project_dir=' + $ProjectExpression + '; cd -- "$project_dir"; ASR_SERVICE_NAME=' + $ServiceLiteral + ' bash scripts/start_asr_server.sh'

if (-not $NoKeepAlive) {
    # A systemd service alone does not keep the WSL VM alive. Task Scheduler keeps
    # this wsl.exe process running; stopping the service ends the loop naturally.
    $Command += '; while :; do state="$(systemctl --user show ' + $ServiceLiteral + ' --property=ActiveState --value)"; case "$state" in active|activating|reloading) sleep 30 ;; inactive|deactivating) exit 0 ;; *) exit 1 ;; esac; done'
}

$WslArguments = @()
if ($Distro) { $WslArguments += @('--distribution', $Distro) }
if ($LinuxUser) { $WslArguments += @('--user', $LinuxUser) }
$WslArguments += @('--exec', 'bash', '-lc', $Command)

& wsl.exe @WslArguments
exit $LASTEXITCODE
