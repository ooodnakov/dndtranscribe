param(
    [Parameter(Mandatory = $true)][string]$RequestPath,
    [Parameter(Mandatory = $true)][string]$ResponsePath,
    [Parameter(Mandatory = $true)][string]$ApiUrl,
    [Parameter(Mandatory = $true)][string]$Model,
    [int]$TimeoutSeconds = 300
)

$ErrorActionPreference = 'Stop'
$lms = Join-Path $env:USERPROFILE '.lmstudio\bin\lms.exe'
if (-not (Test-Path -LiteralPath $lms)) { throw 'LM Studio CLI is not installed' }
$stdout = [System.IO.Path]::GetTempFileName()
$stderr = [System.IO.Path]::GetTempFileName()
try {
    $process = Start-Process -FilePath $lms -ArgumentList @('ps', '--json') -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if ($process.ExitCode -ne 0) {
        $errorOutput = [System.IO.File]::ReadAllText($stderr, [System.Text.Encoding]::UTF8)
        throw "Could not query loaded LM Studio models: $errorOutput"
    }
    $loadedModels = [System.IO.File]::ReadAllText($stdout, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
} finally {
    Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue
}
if (-not ($loadedModels | Where-Object { $_.identifier -eq $Model })) {
    throw "Model '$Model' is not already loaded; start cleanup only after GPU work is idle."
}
$requestBytes = [System.IO.File]::ReadAllBytes($RequestPath)
$response = Invoke-WebRequest `
    -UseBasicParsing `
    -Uri $ApiUrl `
    -Method Post `
    -ContentType 'application/json; charset=utf-8' `
    -Body $requestBytes `
    -TimeoutSec $TimeoutSeconds
$responseBytes = [System.Text.Encoding]::UTF8.GetBytes([string]$response.Content)
[System.IO.File]::WriteAllBytes($ResponsePath, $responseBytes)
