"""Candidate assembly gates and real smoke control flow; no Windows binary required."""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "packaging/windows-smoke.ps1"


def test_manual_candidates_require_both_platforms_without_release_permission() -> None:
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["env"]["BINANCE_AGENT_BUILD_COMMIT"] == "${{ inputs.commit || github.sha }}"
    jobs = workflow["jobs"]
    build = jobs["build-candidates"]
    assert build["if"] == "github.event_name == 'workflow_dispatch'"
    assert build["needs"] == "test"
    targets = {x["target"]: x for x in build["strategy"]["matrix"]["include"]}
    assert set(targets) == {"windows-x64", "macos-arm64"}
    assert targets["macos-arm64"]["architecture"] == "arm64"
    assert targets["windows-x64"]["architecture"] == "x64"
    assert jobs["assemble-candidates"]["needs"] == "build-candidates"
    for name in ("test", "build-candidates", "assemble-candidates"):
        checkout = next(
            s for s in jobs[name]["steps"] if s.get("uses", "").startswith("actions/checkout@")
        )
        assert checkout["with"]["ref"] == "${{ env.BINANCE_AGENT_BUILD_COMMIT }}"
        assert "permissions" not in jobs[name]
    assert "gh release" not in text
    assert "contents: write" not in text


@pytest.mark.parametrize(
    ("scenario", "error"),
    [
        ("success", None),
        ("health_unavailable", "/health did not become healthy"),
        ("wrong_version", "/health did not become healthy"),
        ("wrong_profile", "/health did not become healthy"),
        ("invalid_example", "synthetic example report missing"),
        ("session_error", "session request failed"),
        ("missing_session", "session response missing session_id"),
        ("missing_run", "turn response missing run_id"),
        ("failed_run", "research run failed"),
        ("missing_basis", "completed report missing or malformed"),
        ("malformed_fact", "malformed report evidence"),
        ("wrong_target", "report target mismatch"),
        ("invalid_quote", "fact lacks matching body evidence"),
        ("missing_markdown", "saved report Markdown missing title"),
        ("topic_advice", "independent topic contains trading scope"),
    ],
)
def test_windows_smoke_checks_before_cleanup(
    tmp_path: Path, scenario: str, error: str | None
) -> None:
    pwsh = shutil.which("pwsh")
    if not pwsh:
        pytest.skip("PowerShell unavailable; Windows frozen executable still requires CI")
    events = tmp_path / "events.txt"
    harness = tmp_path / "harness.ps1"
    harness.write_text(
        """
param([string]$Smoke, [string]$Artifacts, [string]$Events, [string]$Scenario)
$ErrorActionPreference = 'Stop'
function Record([string]$Message) { Add-Content $Events $Message }
function Start-Process {
    param($FilePath, [switch]$PassThru, $RedirectStandardError)
    if ($env:PYTHON_DOTENV_DISABLED -ne '1') { throw 'dotenv must stay disabled' }
    Record 'start'
    Set-Content $RedirectStandardError 'diagnostic sentinel'
    return [pscustomobject]@{ Id = 123; HasExited = $false }
}
function Stop-Process { param($Id, [switch]$Force, $ErrorAction); Record 'stop' }
function Start-Sleep { param($Seconds) }
function Invoke-WebRequest {
    param($Uri, $TimeoutSec, $Method, $ContentType, $Body)
    if ($Uri.EndsWith('/health')) {
        Record 'health'
        if ($Scenario -eq 'health_unavailable') { throw 'not ready' }
        $version = if ($Scenario -eq 'wrong_version') { 'old' } else { '0.3.0a1' }
        $profile = if ($Scenario -eq 'wrong_profile') { 'default' } else { 'demo' }
        return @{StatusCode = 200; Content = (@{status='ok'; version=$version; profile=$profile} | ConvertTo-Json)}
    }
    if ($Uri.EndsWith('/reports/example')) {
        Record 'example'
        return @{Content=(@{report=@{synthetic=($Scenario -ne 'invalid_example'); title='Example'; summary='Synthetic'}} | ConvertTo-Json)}
    }
    if ($Uri.EndsWith('/research/sessions') -and $Method -eq 'Post') {
        $body = $Body | ConvertFrom-Json
        $name = if ($body.mode -eq 'report') { 'market' } else { 'topic' }
        Record "create-$name"
        if ($Scenario -eq 'session_error') { throw 'session request failed' }
        $sid = if ($Scenario -eq 'missing_session') { '' } else { "$name-s" }
        return @{Content=(@{session_id=$sid} | ConvertTo-Json)}
    }
    $name = if ($Uri.Contains('topic-')) { 'topic' } else { 'market' }
    if ($Uri.EndsWith('/turns') -and $Method -eq 'Post') {
        Record "turn-$name"
        if (($Body | ConvertFrom-Json).action -ne 'initial') { throw 'expected initial turn' }
        $rid = if ($Scenario -eq 'missing_run') { '' } else { "$name-r" }
        return @{Content=(@{run_id=$rid} | ConvertTo-Json)}
    }
    if ($Uri.EndsWith("/research/sessions/$name-s")) {
        Record "session-$name"
        $status = if ($Scenario -eq 'failed_run') { 'failed' } else { 'completed' }
        return @{Content=(@{runs=@(@{run_id="$name-r";status=$status})} | ConvertTo-Json -Depth 5)}
    }
    if ($Uri.EndsWith('/report.md')) {
        Record "markdown-$name"
        return @{Content=$(if ($Scenario -eq 'missing_markdown') { 'missing' } else { '# Saved report' })}
    }
    if ($Uri.EndsWith('/report')) {
        Record "report-$name"
        $target = if ($name -eq 'topic') { @{kind='independent'} } else { @{kind='asset';symbol='BTCUSDT';market='fapi'} }
        if ($Scenario -eq 'wrong_target') { $target=@{kind='other'} }
        $eid = if ($Scenario -eq 'malformed_fact') { '' } else { 'e1' }
        $basis = @(@{statement='supported';quote='verified quote';evidence_id=$eid})
        if ($Scenario -eq 'missing_basis') { $basis=@() }
        $advice = if ($name -eq 'topic' -and $Scenario -eq 'topic_advice') { @{decision_type='trade'} } else { $null }
        return @{Content=(@{run_id="$name-r";report=@{kind=$name;title='Saved report';summary='Supported';analysis_at='2026-11-14';
            target=$target;basis=$basis;advice=$advice;completeness='complete'}} | ConvertTo-Json -Depth 8)}
    }
    if ($Uri.Contains('/evidence/')) {
        Record "evidence-$name"
        $text = if ($Scenario -eq 'invalid_quote') { 'other text' } else { 'verified quote' }
        return @{Content=(@{level='body';text=$text} | ConvertTo-Json)}
    }
    throw "unexpected request: $Uri"
}
& $Smoke -Executable 'mock.exe' -ArtifactsDirectory $Artifacts -ExpectedVersion '0.3.0a1'
""",
        encoding="utf-8",
    )
    artifacts = tmp_path / "artifacts"
    result = subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(harness),
            "-Smoke",
            str(SCRIPT),
            "-Artifacts",
            str(artifacts),
            "-Events",
            str(events),
            "-Scenario",
            scenario,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    trace = events.read_text().splitlines()
    assert trace[0] == "start"
    assert trace[-1] == "stop", result.stdout + result.stderr
    assert trace.count("stop") == 1
    if error:
        assert result.returncode != 0
        assert error in result.stdout + result.stderr
        assert "diagnostic sentinel" in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert trace == ["start", "health", "example"] + [
            f"{action}-{name}"
            for name in ("market", "topic")
            for action in ("create", "turn", "session", "report", "evidence", "markdown")
        ] + ["stop"]
        for name in ("market", "topic"):
            record = json.loads((artifacts / f"{name}-report.json").read_text(encoding="utf-8-sig"))
            assert record["report"]["kind"] == name
            assert (artifacts / f"{name}-report.md").exists()


@pytest.fixture
def release_tool():
    spec = importlib.util.spec_from_file_location(
        "candidate_release", ROOT / "packaging/release.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "failure", ["missing", "commit", "version", "lock", "checksum", "dirty", "duplicate"]
)
def test_candidate_assembly_rejects_incomplete_or_mixed_inputs(tmp_path, release_tool, failure):
    identity = {
        "version": "0.3.0a1",
        "commit": "abc123",
        "lock_sha256": "lock",
        "source_dirty": False,
    }
    incoming = tmp_path / "incoming"
    for target in release_tool.TARGETS:
        if failure == "missing" and target == "macos-arm64":
            continue
        directory = incoming / target
        directory.mkdir(parents=True)
        archive = directory / f"binance-agent-0.3.0a1-{target}-unsigned.zip"
        archive.write_bytes(b"candidate archive")
        record = {
            **identity,
            "target": target,
            "signed": False,
            "notarized": False,
            "artifact": archive.name,
            "artifact_sha256": release_tool.sha256(archive),
        }
        if target == "macos-arm64":
            field = {
                "commit": "commit",
                "version": "version",
                "lock": "lock_sha256",
                "checksum": "artifact_sha256",
            }.get(failure)
            if field:
                record[field] = "different"
            if failure == "dirty":
                record["source_dirty"] = True
        manifest = archive.with_suffix(".json")
        manifest.write_text(json.dumps(record))
        if failure == "duplicate" and target == "macos-arm64":
            duplicate = incoming / "duplicate"
            duplicate.mkdir()
            shutil.copy2(manifest, duplicate / manifest.name)
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        release_tool.aggregate_candidates(incoming, output, identity)
    assert not output.exists()


def test_candidate_assembly_includes_same_revision_source_and_checksums(tmp_path, release_tool):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    identity = {
        "version": "0.3.0a1",
        "commit": commit,
        "lock_sha256": "test-lock",
        "source_dirty": False,
    }
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    for target in release_tool.TARGETS:
        archive = incoming / f"binance-agent-0.3.0a1-{target}-unsigned.zip"
        archive.write_bytes(target.encode())
        record = {
            **identity,
            "target": target,
            "signed": False,
            "notarized": False,
            "artifact": archive.name,
            "artifact_sha256": release_tool.sha256(archive),
        }
        archive.with_suffix(".json").write_text(json.dumps(record))
    output = tmp_path / "output"
    release_tool.aggregate_candidates(incoming, output, identity)
    assert (output / "binance-agent-0.3.0a1-source.zip").is_file()
    summary = json.loads((output / "candidate.json").read_text())
    assert summary["commit"] == commit
    assert summary["targets"] == ["macos-arm64", "windows-x64"]
    for line in (output / "SHA256SUMS").read_text().splitlines():
        checksum, filename = line.split("  ", 1)
        assert release_tool.sha256(output / filename) == checksum


@pytest.mark.parametrize(
    "scenario", ["success", "missing_run", "invalid_quote", "process_exit", "topic_advice"]
)
def test_macos_smoke_validates_evidence_and_cleans_its_process(tmp_path, monkeypatch, scenario):
    monkeypatch.syspath_prepend(str(ROOT / "packaging"))
    spec = importlib.util.spec_from_file_location("macos_smoke", ROOT / "packaging/macos-smoke.py")
    assert spec and spec.loader
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    monkeypatch.setattr(smoke, "release_version", lambda: "0.3.0a1")
    events = []

    class Process:
        returncode = 9 if scenario == "process_exit" else None

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append("terminate")
            self.returncode = 0

        def wait(self, timeout):
            events.append("wait")
            return self.returncode

    process = Process()

    def launch(args, **kwargs):
        assert kwargs["env"]["BINANCE_AGENT_PROFILE"] == "demo"
        assert kwargs["env"]["PYTHON_DOTENV_DISABLED"] == "1"
        assert Path(kwargs["env"]["BINANCE_AGENT_DATA_DIR"]).is_relative_to(tmp_path)
        assert Path(kwargs["env"]["BINANCE_AGENT_RUNTIME_CONFIG"]).is_relative_to(tmp_path)
        assert kwargs["cwd"] == tmp_path
        events.append("start")
        return process

    class Opener:
        def open(self, req, timeout):
            url = req.full_url
            if url.endswith("/health"):
                value = {"status": "ok", "profile": "demo", "version": "0.3.0a1"}
            elif url.endswith("/reports/example"):
                value = {"report": {"synthetic": True, "title": "Example", "summary": "Synthetic"}}
            elif url.endswith("/research/sessions"):
                body = json.loads(req.data)
                name = "market" if body["mode"] == "report" else "topic"
                value = {"session_id": f"{name}-s"}
            else:
                name = "topic" if "topic-" in url else "market"
                if url.endswith("/turns"):
                    assert json.loads(req.data)["action"] == "initial"
                    value = {} if scenario == "missing_run" else {"run_id": f"{name}-r"}
                elif url.endswith(f"/research/sessions/{name}-s"):
                    value = {"runs": [{"run_id": f"{name}-r", "status": "completed"}]}
                elif "/evidence/" in url:
                    events.append("evidence")
                    value = {
                        "level": "body",
                        "text": "other" if scenario == "invalid_quote" else "verified quote",
                    }
                elif url.endswith("/report.md"):
                    return io.BytesIO(b"# Saved report")
                elif url.endswith("/report"):
                    target = (
                        {"kind": "independent"}
                        if name == "topic"
                        else {"kind": "asset", "symbol": "BTCUSDT", "market": "fapi"}
                    )
                    value = {
                        "run_id": f"{name}-r",
                        "report": {
                            "kind": name,
                            "title": "Saved report",
                            "summary": "Supported",
                            "analysis_at": "2026-11-14",
                            "target": target,
                            "basis": [
                                {
                                    "statement": "supported",
                                    "quote": "verified quote",
                                    "evidence_id": "e1",
                                }
                            ],
                            "completeness": "complete",
                            "advice": {}
                            if name == "topic" and scenario == "topic_advice"
                            else None,
                        },
                    }
                else:
                    raise AssertionError(f"unexpected request: {url}")
            return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(smoke.subprocess, "Popen", launch)
    monkeypatch.setattr(smoke.urllib.request, "build_opener", lambda *args: Opener())
    if scenario == "success":
        smoke.smoke(tmp_path / "mock-executable", tmp_path)
        assert events == ["start", "evidence", "evidence", "terminate", "wait"]
    else:
        with pytest.raises(RuntimeError):
            smoke.smoke(tmp_path / "mock-executable", tmp_path)
        if scenario != "process_exit":
            assert events[-2:] == ["terminate", "wait"]


def test_release_version_rejects_source_package_mismatch(tmp_path, release_tool):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.3.0a1"\n')
    source = tmp_path / "src/binance_agent"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text('__version__ = "0.2.0"\n')
    with pytest.raises(ValueError, match="source version"):
        release_tool.release_version(tmp_path)
    (source / "__init__.py").write_text('__version__ = "0.3.0a1"\n')
    assert release_tool.release_version(tmp_path) == "0.3.0a1"


@pytest.mark.parametrize("target", ["windows-x64", "macos-arm64"])
def test_platform_archive_preserves_bundle_and_matching_manifest(
    tmp_path, release_tool, monkeypatch, target
):
    if target == "macos-arm64" and sys.platform != "darwin":
        pytest.skip("ditto archive verification requires macOS")
    identity = {
        "version": "0.3.0a1",
        "commit": "abc123",
        "lock_sha256": "lock",
        "source_dirty": False,
    }
    monkeypatch.setattr(release_tool, "source_identity", lambda: identity)
    monkeypatch.setattr(release_tool, "check_target", lambda target: None)
    dist = tmp_path / "dist"
    bundle = dist / ("Binance Agent.app" if target == "macos-arm64" else "binance-agent")
    licenses = bundle / "third-party-licenses"
    licenses.mkdir(parents=True)
    for name in ["feedparser-LICENSE", "feedparser-sgmllib-LICENSE", "sources.json"]:
        (licenses / name).write_text("test fixture")
    (bundle / "bundle-build.json").write_text(json.dumps(identity))
    (bundle / "payload.txt").write_text("application fixture")
    if target == "macos-arm64":
        (bundle / "payload-link").symlink_to("payload.txt")
    manifest = release_tool.package_bundle(target, dist, tmp_path / "candidate")
    record = json.loads(manifest.read_text())
    archive = manifest.parent / record["artifact"]
    assert record["artifact_sha256"] == release_tool.sha256(archive)
    assert record["target"] == target
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.read(f"{bundle.name}/payload.txt") == b"application fixture"
        if target == "macos-arm64":
            assert (
                zipped.getinfo(f"{bundle.name}/payload-link").external_attr >> 16 & 0o170000
                == 0o120000
            )


def test_bundle_inventory_rejects_runtime_configuration(tmp_path, release_tool):
    (tmp_path / "runtime_config.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected runtime"):
        release_tool.validate_bundle(tmp_path)


def test_bundle_distribution_names_match_locked_hyphenated_names(release_tool):
    actual = release_tool.verify_dependencies(
        {"jaraco.classes", "jaraco.context", "jaraco.functools"}
    )
    assert {"jaraco-classes", "jaraco-context", "jaraco-functools"} <= actual.keys()
    assert all(actual[name] in release_tool.locked_versions()[name] for name in actual)
