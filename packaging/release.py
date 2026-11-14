"""Build metadata and assemble an unsigned candidate; never publish a Release."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"windows-x64": ("Windows", "amd64"), "macos-arm64": ("Darwin", "arm64")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_version(root: Path = ROOT) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    tree = ast.parse((root / "src/binance_agent/__init__.py").read_text(encoding="utf-8"))
    actual = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
    )
    if actual != version:
        raise ValueError(f"source version {actual} != project version {version}")
    return str(version)


def normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_versions(root: Path = ROOT) -> dict[str, set[str]]:
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    result: dict[str, set[str]] = {}
    for package in lock["package"]:
        result.setdefault(normalized_name(package["name"]), set()).add(package["version"])
    return result


def verify_dependencies(names: set[str], root: Path = ROOT) -> dict[str, str]:
    locked = locked_versions(root)
    if "sgmllib3k" in locked:
        raise ValueError("uv.lock still includes retired sgmllib3k")
    try:
        metadata.distribution("sgmllib3k")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise ValueError("build environment still includes retired sgmllib3k; use uv sync --locked")
    actual = {}
    for name in sorted(names | {"feedparser", "feedparser-sgmllib"}):
        normalized = normalized_name(name)
        version = metadata.version(name)
        if version not in locked.get(normalized, set()):
            raise ValueError(f"{name} {version} does not match uv.lock")
        actual[normalized] = version
    if sys.platform == "darwin":
        for name, version in actual.items():
            if name.startswith("pyobjc-") and version != "12.2.2":
                raise ValueError(f"{name} must use the licensed PyObjC 12.2.2 baseline")
    return actual


def source_identity(root: Path = ROOT) -> dict[str, object]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    expected = os.environ.get("BINANCE_AGENT_BUILD_COMMIT") or os.environ.get("GITHUB_SHA")
    if expected and expected.lower() != commit.lower():
        raise ValueError("checkout does not match requested build commit")
    changed = (
        subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=root, check=False).returncode != 0
    )
    untracked = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    generated = {
        "dist",
        "build",
        "smoke-artifacts",
        "candidate-artifacts",
        "incoming",
        "preview-candidate",
    }
    unpublished = any(name and Path(name).parts[0] not in generated for name in untracked)
    return {
        "version": release_version(root),
        "commit": commit,
        "lock_sha256": sha256(root / "uv.lock"),
        "source_dirty": changed or unpublished,
    }


def check_target(target: str) -> None:
    system, machine = TARGETS[target]
    if platform.system() != system or platform.machine().lower() != machine:
        raise ValueError(
            f"{target} requires {system}/{machine}, got {platform.system()}/{platform.machine()}"
        )


def validate_bundle(bundle: Path) -> dict[str, object]:
    forbidden = {".env", "secrets.enc", "runtime_config.json", ".runs", ".demo", ".binance_agent"}
    for path in bundle.rglob("*"):
        if (
            path.name in forbidden
            or path.name.startswith(".env.")
            or "sgmllib3k" in path.name.lower()
        ):
            raise ValueError(
                f"unexpected runtime/retired dependency path: {path.relative_to(bundle)}"
            )
    manifests = list(bundle.rglob("bundle-build.json"))
    if not manifests:
        raise ValueError("bundle-build.json missing from bundle")
    records = [json.loads(p.read_text(encoding="utf-8")) for p in manifests]
    if any(record != records[0] for record in records[1:]):
        raise ValueError("bundle metadata copies disagree")
    record = records[0]
    for key, value in source_identity().items():
        if record.get(key) != value:
            raise ValueError(f"bundle {key} does not match current source")
    for filename in ["feedparser-LICENSE", "feedparser-sgmllib-LICENSE", "sources.json"]:
        if not list(bundle.rglob(filename)):
            raise ValueError(f"required license missing: {filename}")
    return record


def package_bundle(target: str, dist: Path, output: Path) -> Path:
    check_target(target)
    bundle = dist / ("Binance Agent.app" if target == "macos-arm64" else "binance-agent")
    record = validate_bundle(bundle)
    output.mkdir(parents=True, exist_ok=True)
    basename = f"binance-agent-{record['version']}-{target}-unsigned"
    archive = output / f"{basename}.zip"
    if target == "macos-arm64":
        subprocess.run(
            ["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(bundle), str(archive)],
            check=True,
        )
    else:
        shutil.make_archive(str(output / basename), "zip", root_dir=dist, base_dir=bundle.name)
    record.update(
        target=target,
        signed=False,
        notarized=False,
        build_system=platform.platform(),
        artifact=archive.name,
        artifact_sha256=sha256(archive),
    )
    manifest = output / f"{basename}.json"
    manifest.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def aggregate_candidates(incoming: Path, output: Path, identity: dict[str, object]) -> None:
    """Fail closed on a missing platform, mixed revision, or changed artifact."""
    if identity.get("source_dirty") is not False:
        raise ValueError("candidate assembly requires a clean committed source snapshot")
    manifests = list(incoming.rglob("binance-agent-*-unsigned.json"))
    found = {}
    for manifest in manifests:
        record = json.loads(manifest.read_text(encoding="utf-8"))
        target = record.get("target")
        if target not in TARGETS or target in found:
            raise ValueError("unexpected or duplicate candidate platform")
        if any(record.get(k) != v for k, v in identity.items()):
            raise ValueError("candidate version/commit/lock mismatch")
        if record.get("signed") is not False or record.get("notarized") is not False:
            raise ValueError("candidate does not match unsigned preview policy")
        expected = f"binance-agent-{identity['version']}-{target}-unsigned.zip"
        if record.get("artifact") != expected:
            raise ValueError("unexpected artifact name")
        archive = manifest.parent / expected
        if not archive.is_file() or sha256(archive) != record.get("artifact_sha256"):
            raise ValueError("candidate archive missing or checksum mismatch")
        found[target] = (manifest, archive)
    if set(found) != set(TARGETS):
        raise ValueError("both windows-x64 and macos-arm64 candidates are required")
    if output.exists() and any(output.iterdir()):
        raise ValueError("assembly output must be empty to avoid mixing old candidate files")
    output.mkdir(parents=True, exist_ok=True)
    for manifest, archive in found.values():
        shutil.copy2(manifest, output / manifest.name)
        shutil.copy2(archive, output / archive.name)
    source = output / f"binance-agent-{identity['version']}-source.zip"
    subprocess.run(
        [
            "git",
            "archive",
            "--format=zip",
            f"--prefix=binance-agent-{identity['version']}/",
            f"--output={source.resolve()}",
            str(identity["commit"]),
        ],
        cwd=ROOT,
        check=True,
    )
    summary = {
        **identity,
        "signed": False,
        "notarized": False,
        "targets": sorted(found),
        "source": source.name,
    }
    (output / "candidate.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    files = sorted(p for p in output.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in files), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--target", choices=TARGETS, required=True)
    package = commands.add_parser("package")
    package.add_argument("--target", choices=TARGETS, required=True)
    package.add_argument("--dist", type=Path, default=ROOT / "dist")
    package.add_argument("--output", type=Path, required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--incoming", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        check_target(args.target)
        verify_dependencies({"pyobjc-core"} if args.target == "macos-arm64" else set())
        print(json.dumps(source_identity(), indent=2))
    elif args.command == "package":
        print(package_bundle(args.target, args.dist, args.output))
    else:
        aggregate_candidates(args.incoming, args.output, source_identity())


if __name__ == "__main__":
    main()
