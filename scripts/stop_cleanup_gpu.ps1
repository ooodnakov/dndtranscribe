$ErrorActionPreference = 'Stop'

$configPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\work\llm_config.json'))
if (-not (Test-Path -LiteralPath $configPath)) { throw "LLM config not found at $configPath" }
$config = [System.IO.File]::ReadAllText($configPath, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
$model = [string]$config.model
$lms = Join-Path $env:USERPROFILE '.lmstudio\bin\lms.exe'
if (-not (Test-Path -LiteralPath $lms)) { throw "LM Studio CLI not found at $lms" }

function Invoke-Lms([string[]]$Arguments) {
    $stdout = [System.IO.Path]::GetTempFileName()
    $stderr = [System.IO.Path]::GetTempFileName()
    try {
        $process = Start-Process -FilePath $lms -ArgumentList $Arguments -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $output = [System.IO.File]::ReadAllText($stdout, [System.Text.Encoding]::UTF8)
        $errorOutput = [System.IO.File]::ReadAllText($stderr, [System.Text.Encoding]::UTF8)
        if ($process.ExitCode -ne 0) { throw "LM Studio CLI failed: $errorOutput$output" }
        return $output
    } finally {
        Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue
    }
}

$loaded = (Invoke-Lms @('ps', '--json')) | ConvertFrom-Json
if ($loaded | Where-Object { $_.identifier -eq $model }) {
    Invoke-Lms @('unload', $model) | Out-Null
}

$remaining = (Invoke-Lms @('ps', '--json')) | ConvertFrom-Json
if ($remaining | Where-Object { $_.identifier -eq $model }) {
    throw 'Gemma4 is still loaded; do not resume GPU diarization yet'
}
Write-Output "Unloaded $model; LM Studio API remains available without a model."
