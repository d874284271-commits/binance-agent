param(
    [string]$Executable = "$PSScriptRoot/../dist/binance-agent/binance-agent.exe",
    [string]$ArtifactsDirectory = "$PSScriptRoot/../smoke-artifacts",
    [int]$Port = 0,
    [string]$ExpectedVersion = ''
)

$ErrorActionPreference = 'Stop'
if (-not $ExpectedVersion) {
    $ExpectedVersion = (Select-String -Path "$PSScriptRoot/../pyproject.toml" -Pattern '^version = "([^"]+)"').Matches.Groups[1].Value
}
if ($Port -eq 0) {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    try { $Port = $listener.LocalEndpoint.Port } finally { $listener.Stop() }
}
New-Item -ItemType Directory -Path $ArtifactsDirectory -Force | Out-Null
$ArtifactsDirectory = (Resolve-Path $ArtifactsDirectory).Path
# A unique data directory prevents a previous run from satisfying this smoke test.
$env:BINANCE_AGENT_DATA_DIR = Join-Path $ArtifactsDirectory ([guid]::NewGuid().ToString())
$env:BINANCE_AGENT_HEADLESS = '1'
$env:BINANCE_AGENT_PROFILE = 'demo'
$env:PYTHON_DOTENV_DISABLED = '1'
$env:BINANCE_AGENT_CONFIG = Join-Path $ArtifactsDirectory 'research-smoke.toml'
$env:BINANCE_AGENT_RUNTIME_CONFIG = Join-Path $env:BINANCE_AGENT_DATA_DIR 'runtime_config.json'
@"
[app]
port = $Port
[research]
enabled = true
max_requests = 100
deadline_s = 600
input_mode = "evidence"
workflow = "multi_debate"
[collaboration]
enabled = true
"@ | Set-Content -Path $env:BINANCE_AGENT_CONFIG -Encoding utf8
$stderrPath = Join-Path $ArtifactsDirectory 'app-stderr.log'
$proc = $null
$baseUri = "http://127.0.0.1:$Port"
try {
    # Leave stdout unredirected to exercise the windowed executable's stream fallback.
    $proc = Start-Process -FilePath $Executable -PassThru -RedirectStandardError $stderrPath
    $healthy = $false
    foreach ($attempt in 1..40) {
        Start-Sleep -Seconds 3
        if ($proc.HasExited) { break }
        try {
            $response = Invoke-WebRequest -Uri "$baseUri/health" -TimeoutSec 5
            $health = $response.Content | ConvertFrom-Json
            if ($response.StatusCode -eq 200 -and $health.status -eq 'ok' -and
                $health.version -eq $ExpectedVersion -and $health.profile -eq 'demo') {
                $healthy = $true
                break
            }
        } catch { }
    }
    if (-not $healthy) { throw 'smoke failed: /health did not become healthy within 120s' }

    $example = (Invoke-WebRequest -Uri "$baseUri/reports/example" -TimeoutSec 15).Content | ConvertFrom-Json
    if ($example.report.synthetic -ne $true -or -not $example.report.title -or -not $example.report.summary) {
        throw 'smoke failed: synthetic example report missing'
    }
    $cases = @(
        @{name='market'; mode='report'; target=@{kind='asset'; symbol='BTCUSDT'; market='fapi'}; question=''},
        @{name='topic'; mode='topic'; target=@{kind='independent'}; question='Digital asset regulation proposal'}
    )
    foreach ($case in $cases) {
        $body = @{mode=$case.mode; target=$case.target; question=$case.question; request_id=[guid]::NewGuid().ToString()} | ConvertTo-Json -Depth 5
        $session = (Invoke-WebRequest -Uri "$baseUri/research/sessions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 15).Content | ConvertFrom-Json
        if (-not $session.session_id) { throw 'smoke failed: session response missing session_id' }
        $sessionId = [uri]::EscapeDataString($session.session_id)
        $body = @{action='initial'; request_id=[guid]::NewGuid().ToString()} | ConvertTo-Json
        $turn = (Invoke-WebRequest -Uri "$baseUri/research/sessions/$sessionId/turns" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 15).Content | ConvertFrom-Json
        if (-not $turn.run_id) { throw 'smoke failed: turn response missing run_id' }
        $runId = [uri]::EscapeDataString($turn.run_id)
        $run = $null
        foreach ($attempt in 1..60) {
            if ($proc.HasExited) { throw 'smoke failed: app exited during research' }
            $response = Invoke-WebRequest -Uri "$baseUri/research/sessions/$sessionId" -TimeoutSec 15
            $detail = $response.Content | ConvertFrom-Json
            $run = $detail.runs | Where-Object { $_.run_id -eq $turn.run_id } | Select-Object -First 1
            if ($run.status -eq 'completed') { break }
            if ($run.status -in @('failed', 'cancelled', 'interrupted')) { throw 'smoke failed: research run failed' }
            Start-Sleep -Seconds 3
        }
        $response.Content | Set-Content (Join-Path $ArtifactsDirectory "$($case.name)-session.json")
        if (-not $run -or $run.status -ne 'completed') { throw 'smoke failed: research run did not complete' }
        $response = Invoke-WebRequest -Uri "$baseUri/runs/$runId/report" -TimeoutSec 15
        $response.Content | Set-Content (Join-Path $ArtifactsDirectory "$($case.name)-report.json")
        $result = $response.Content | ConvertFrom-Json
        $report = $result.report
        if ($result.run_id -ne $turn.run_id -or $report.kind -ne $case.name -or
            -not $report.title -or -not $report.summary -or -not $report.analysis_at -or
            $report.completeness -notin @('complete', 'partial') -or -not $report.basis) {
            throw 'smoke failed: completed report missing or malformed'
        }
        if ($report.target.kind -ne $case.target.kind) { throw 'smoke failed: report target mismatch' }
        if ($case.name -eq 'market' -and ($report.target.symbol -ne 'BTCUSDT' -or $report.target.market -ne 'fapi')) {
            throw 'smoke failed: market report target mismatch'
        }
        if ($case.name -eq 'topic' -and ($null -ne $report.advice -or $report.target.symbol)) {
            throw 'smoke failed: independent topic contains trading scope'
        }
        foreach ($fact in $report.basis) {
            if (-not $fact.statement -or -not $fact.quote -or -not $fact.evidence_id) { throw 'smoke failed: malformed report evidence' }
            $evidenceId = [uri]::EscapeDataString($fact.evidence_id)
            $document = (Invoke-WebRequest -Uri "$baseUri/runs/$runId/evidence/$evidenceId" -TimeoutSec 15).Content | ConvertFrom-Json
            if ($document.level -ne 'body' -or -not $document.text.Contains($fact.quote)) {
                throw 'smoke failed: fact lacks matching body evidence'
            }
        }
        $markdown = (Invoke-WebRequest -Uri "$baseUri/runs/$runId/report.md" -TimeoutSec 15).Content
        if (-not $markdown.Contains($report.title)) { throw 'smoke failed: saved report Markdown missing title' }
        $markdown | Set-Content (Join-Path $ArtifactsDirectory "$($case.name)-report.md")
    }
    Write-Host 'offline market and independent-topic report smoke passed'

} catch {
    Write-Host '::group::smoke diagnostics'
    Get-Content $stderrPath -ErrorAction SilentlyContinue | Write-Host
    Get-ChildItem $env:BINANCE_AGENT_DATA_DIR -Recurse -Filter '*.log' -ErrorAction SilentlyContinue |
        ForEach-Object { Get-Content $_.FullName -Tail 100 -ErrorAction SilentlyContinue | Write-Host }
    Write-Host '::endgroup::'
    throw
} finally {
    if ($proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
}
