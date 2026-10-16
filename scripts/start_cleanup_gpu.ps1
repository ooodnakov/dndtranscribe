$ErrorActionPreference = 'Stop'

$configPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\work\llm_config.json'))
if (-not (Test-Path -LiteralPath $configPath)) { throw "LLM config not found at $configPath" }
$config = [System.IO.File]::ReadAllText($configPath, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
$model = [string]$config.model
$uri = [System.Uri]$config.api_url
if (-not $uri.IsLoopback) { throw 'The LM Studio API must bind to localhost' }
$port = $uri.Port
if (-not $model -or -not $port) { throw 'LLM config must include model and api_url' }
$lms = Join-Path $env:USERPROFILE '.lmstudio\bin\lms.exe'
if (-not (Test-Path -LiteralPath $lms)) {
    throw "LM Studio CLI not found at $lms"
}

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

$status = Invoke-Lms @('status')
if ($status -match 'Server:\s+OFF') {
    Invoke-Lms @('server', 'start', '--port', [string]$port, '--bind', '127.0.0.1') | Out-Null
} elseif ($status -notmatch "Server:\s+ON \(port: $port\)") {
    throw "LM Studio is serving on an unexpected port: $status"
}

Invoke-Lms @('load', $model, '--gpu', 'max', '--context-length', '4096', '--parallel', '1', '--yes') | Out-Null

$loaded = (Invoke-Lms @('ps', '--json')) | ConvertFrom-Json
if (-not ($loaded | Where-Object { $_.identifier -eq $model })) {
    throw 'Gemma4 is not listed as loaded after the load command'
}
Write-Output "Loaded $model through LM Studio with full GPU offload and context length 4096."
