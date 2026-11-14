"""Run an isolated, offline macOS bundle smoke; native UI acceptance is separate."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from release import release_version


def smoke(executable: Path, artifacts: Path) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    data = Path(tempfile.mkdtemp(prefix="data-", dir=artifacts))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    config = artifacts / "research-smoke.toml"
    config.write_text(
        f"[app]\nport = {port}\n[research]\nenabled = true\nmax_requests = 100\n"
        'deadline_s = 600\ninput_mode = "evidence"\nworkflow = "multi_debate"\n[collaboration]\nenabled = true\n',
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "BINANCE_AGENT_HEADLESS": "1",
        "BINANCE_AGENT_PROFILE": "demo",
        "PYTHON_DOTENV_DISABLED": "1",
        "BINANCE_AGENT_DATA_DIR": str(data),
        "BINANCE_AGENT_RUNTIME_CONFIG": str(data / "runtime_config.json"),
        "BINANCE_AGENT_CONFIG": str(config),
    }
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path: str, body: dict | None = None, timeout: int = 15) -> str:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        with opener.open(req, timeout=timeout) as response:
            return response.read().decode("utf-8")

    with (artifacts / "app.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(executable)], cwd=artifacts, env=env, stdout=log, stderr=log
        )
        try:
            expected_version = release_version()
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f"bundle exited before health: {process.returncode}")
                try:
                    health = json.loads(request("/health", timeout=2))
                    if (
                        health.get("status") == "ok"
                        and health.get("profile") == "demo"
                        and health.get("version") == expected_version
                    ):
                        break
                except (OSError, ValueError, urllib.error.URLError):
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("matching demo /health did not become healthy")
            example = json.loads(request("/reports/example")).get("report") or {}
            if (
                example.get("synthetic") is not True
                or not example.get("title")
                or not example.get("summary")
            ):
                raise RuntimeError("synthetic example report missing")
            cases = [
                ("market", "report", {"kind": "asset", "symbol": "BTCUSDT", "market": "fapi"}, ""),
                ("topic", "topic", {"kind": "independent"}, "Digital asset regulation proposal"),
            ]
            for name, mode, target, question in cases:
                session = json.loads(
                    request(
                        "/research/sessions",
                        {
                            "mode": mode,
                            "target": target,
                            "question": question,
                            "request_id": uuid.uuid4().hex,
                        },
                    )
                )
                if not session.get("session_id"):
                    raise RuntimeError("session response missing session_id")
                sid = urllib.parse.quote(session["session_id"], safe="")
                turn = json.loads(
                    request(
                        f"/research/sessions/{sid}/turns",
                        {
                            "action": "initial",
                            "request_id": uuid.uuid4().hex,
                        },
                    )
                )
                if not turn.get("run_id"):
                    raise RuntimeError("turn response missing run_id")
                run_id = urllib.parse.quote(turn["run_id"], safe="")
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("app exited during research")
                    detail = json.loads(request(f"/research/sessions/{sid}"))
                    run = next(
                        (r for r in detail.get("runs", []) if r.get("run_id") == turn["run_id"]), {}
                    )
                    if run.get("status") == "completed":
                        break
                    if run.get("status") in {"failed", "cancelled", "interrupted"}:
                        raise RuntimeError("research run failed")
                    time.sleep(1)
                else:
                    raise RuntimeError("research run did not complete")
                (artifacts / f"{name}-session.json").write_text(
                    json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                result = json.loads(request(f"/runs/{run_id}/report"))
                (artifacts / f"{name}-report.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                report = result.get("report") or {}
                if (
                    result.get("run_id") != turn["run_id"]
                    or report.get("kind") != name
                    or not all(report.get(k) for k in ("title", "summary", "analysis_at", "basis"))
                    or report.get("completeness") not in {"complete", "partial"}
                ):
                    raise RuntimeError("completed report missing or malformed")
                if report.get("target") != target:
                    raise RuntimeError("report target mismatch")
                if name == "topic" and report.get("advice") is not None:
                    raise RuntimeError("independent topic contains trading scope")
                for fact in report["basis"]:
                    if not all(fact.get(k) for k in ("statement", "quote", "evidence_id")):
                        raise RuntimeError("malformed report evidence")
                    eid = urllib.parse.quote(fact["evidence_id"], safe="")
                    document = json.loads(request(f"/runs/{run_id}/evidence/{eid}"))
                    if document.get("level") != "body" or fact["quote"] not in document.get(
                        "text", ""
                    ):
                        raise RuntimeError("fact lacks matching body evidence")
                markdown = request(f"/runs/{run_id}/report.md")
                if report["title"] not in markdown:
                    raise RuntimeError("saved report Markdown missing title")
                (artifacts / f"{name}-report.md").write_text(markdown, encoding="utf-8")
            print("offline market and independent-topic report smoke passed")

        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("dist/Binance Agent.app/Contents/MacOS/binance-agent"),
    )
    parser.add_argument("--artifacts", type=Path, default=Path("smoke-artifacts"))
    args = parser.parse_args()
    smoke(args.executable.resolve(), args.artifacts.resolve())
