#!/usr/bin/env python3
"""Fail closed on unexpected Schemen Gate distribution contents."""

from __future__ import annotations

import ast
import base64
import csv
import gzip
import hashlib
import io

# Git inspection uses a fixed executable, argument vectors, and no shell.
import subprocess  # nosec B404
import sys
import tarfile
import tempfile
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import BinaryIO

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
EXPECTED_VERSION = "1.0.2"
EXPECTED_REPOSITORY = "https://github.com/sekosai/schemen-gate"
DIST_INFO = f"schemen_gate-{EXPECTED_VERSION}.dist-info"
SDIST_ROOT = f"schemen_gate-{EXPECTED_VERSION}"
EXPECTED_REQUIRES_DIST = {
    "numpy>=1.24",
    'cryptography>=50.0; extra == "crypto"',
    'torch>=2.13; extra == "torch"',
    'cryptography>=50.0; extra == "lockbox"',
    'pyyaml>=6.0; extra == "lockbox"',
    'onnx>=1.14; extra == "onnx"',
    'spiffe>=0.3.0; extra == "spiffe"',
    'psycopg[binary]>=3.1; extra == "rag"',
    'scikit-learn>=1.3; extra == "rag"',
    'mypy<2,>=1.19; extra == "dev"',
    'pypdf>=6.0; extra == "dev"',
    'pytest>=8.0; extra == "dev"',
    'pytest-cov>=6.0; extra == "dev"',
    'ruff>=0.12; extra == "dev"',
    'twine==7.0.0; extra == "dev"',
    'types-PyYAML>=6.0; extra == "dev"',
}
EXPECTED_EXTRAS = {"crypto", "torch", "lockbox", "onnx", "spiffe", "rag", "dev"}
GENERATED_SDIST_FILES = {
    "PKG-INFO",
    "setup.cfg",
    "src/schemen_gate.egg-info/PKG-INFO",
    "src/schemen_gate.egg-info/SOURCES.txt",
    "src/schemen_gate.egg-info/dependency_links.txt",
    "src/schemen_gate.egg-info/requires.txt",
    "src/schemen_gate.egg-info/top_level.txt",
}
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_MEMBER_BYTES = 2 * 1024 * 1024
MAX_TOTAL_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TAR_STREAM_BYTES = 32 * 1024 * 1024
MAX_COMPRESSION_RATIO = 100
COMPRESSION_RATIO_FLOOR = 64 * 1024
READ_CHUNK_BYTES = 64 * 1024


def fail(message: str) -> None:
    raise SystemExit(f"distribution verification failed: {message}")


def require_exact_members(*, actual: set[str], expected: set[str], label: str) -> None:
    if actual != expected:
        fail(
            f"{label} member allowlist mismatch; "
            f"missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}"
        )


def archive_size(path: Path, *, label: str) -> int:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} is not a regular file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > MAX_ARCHIVE_BYTES:
        fail(
            f"{label} compressed size is outside the release bound: "
            f"{size} bytes (maximum {MAX_ARCHIVE_BYTES})"
        )
    return size


def validate_compressed_member(*, declared_size: int, compressed_size: int, label: str) -> None:
    if declared_size < 0 or compressed_size < 0:
        fail(f"{label} has a negative declared size")
    if declared_size > MAX_MEMBER_BYTES:
        fail(
            f"{label} exceeds the per-member release bound: "
            f"{declared_size} bytes (maximum {MAX_MEMBER_BYTES})"
        )
    ratio_bound = max(
        COMPRESSION_RATIO_FLOOR,
        compressed_size * MAX_COMPRESSION_RATIO,
    )
    if declared_size > ratio_bound:
        fail(
            f"{label} exceeds the compression-ratio release bound: "
            f"{declared_size} uncompressed bytes from {compressed_size} compressed bytes"
        )


def read_bounded(stream: BinaryIO, *, declared_size: int, label: str) -> bytes:
    """Read at most one byte beyond a preflighted declaration."""
    if declared_size < 0 or declared_size > MAX_MEMBER_BYTES:
        fail(f"{label} has an invalid declared size: {declared_size}")
    payload = bytearray()
    while True:
        remaining = declared_size + 1 - len(payload)
        if remaining <= 0:
            fail(f"{label} expanded beyond its declared size")
        try:
            chunk = stream.read(min(READ_CHUNK_BYTES, remaining))
        except Exception as exc:
            fail(f"cannot read {label}: {exc}")
        if not chunk:
            break
        payload.extend(chunk)
    if len(payload) != declared_size:
        fail(f"{label} size mismatch: declared {declared_size}, read {len(payload)}")
    return bytes(payload)


def member_fact(payload: bytes) -> tuple[bytes, int]:
    return hashlib.sha256(payload).digest(), len(payload)


def decompress_sdist(path: Path, *, compressed_size: int) -> BinaryIO:
    """Bound gzip expansion before tar metadata is allowed to parse it."""
    spool = tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b")
    total = 0
    try:
        with path.open("rb") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as source:
            while chunk := source.read(READ_CHUNK_BYTES):
                total += len(chunk)
                if total > MAX_TAR_STREAM_BYTES:
                    spool.close()
                    fail(
                        "sdist gzip stream exceeds the release bound: "
                        f"maximum {MAX_TAR_STREAM_BYTES} bytes"
                    )
                spool.write(chunk)
    except (gzip.BadGzipFile, EOFError, OSError) as exc:
        spool.close()
        fail(f"cannot decompress sdist safely: {exc}")
    ratio_bound = max(
        COMPRESSION_RATIO_FLOOR,
        compressed_size * MAX_COMPRESSION_RATIO,
    )
    if total > ratio_bound:
        spool.close()
        fail(
            "sdist exceeds the compression-ratio release bound: "
            f"{total} uncompressed bytes from {compressed_size} compressed bytes"
        )
    spool.seek(0)
    return spool


def git(*arguments: str, text: bool = False) -> bytes | str:
    try:
        output = subprocess.run(  # nosec B603
            ("git", *arguments),
            cwd=ROOT,
            check=True,
            capture_output=True,
            shell=False,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"cannot inspect the reviewed Git commit: {exc}")
    return output.decode("utf-8") if text else output


def expected_commit() -> str:
    value = str(git("rev-parse", "HEAD", text=True)).strip()
    if len(value) not in {40, 64} or any(c not in "0123456789abcdef" for c in value):
        fail(f"source commit is not a canonical Git SHA: {value!r}")
    return value


def validate_archive_path(path: str, *, label: str, allow_root: bool = False) -> None:
    if "\\" in path or "\x00" in path:
        fail(f"{label} is not canonical: {path!r}")
    pure = PurePosixPath(path)
    if (not allow_root and not path) or pure.is_absolute() or ".." in pure.parts:
        fail(f"{label} is unsafe: {path!r}")
    if path not in {"", "."} and pure.as_posix() != path.rstrip("/"):
        fail(f"{label} is not normalized: {path!r}")


def tracked_paths() -> set[str]:
    output = bytes(git("ls-tree", "-r", "-z", "--name-only", "HEAD"))
    paths = {raw.decode("utf-8") for raw in output.split(b"\0") if raw}
    for path in paths:
        validate_archive_path(path, label="Git path")
    return paths


def git_bytes(path: str) -> bytes:
    return bytes(git("show", f"HEAD:{path}"))


def parse_build_identity(source: bytes, *, archive_name: str) -> dict[str, str]:
    try:
        tree = ast.parse(source.decode("utf-8"), filename=archive_name)
    except (SyntaxError, UnicodeDecodeError) as exc:
        fail(f"invalid build identity in {archive_name}: {exc}")
    values: dict[str, str] = {}
    allowed = {"SOURCE_VERSION", "SOURCE_REPOSITORY", "SOURCE_COMMIT"}
    for statement in tree.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
            or statement.targets[0].id not in allowed
            or not isinstance(statement.value, ast.Constant)
            or not isinstance(statement.value.value, str)
        ):
            fail(f"unexpected executable content in {archive_name}")
        name = statement.targets[0].id
        if name in values:
            fail(f"duplicate {name} in {archive_name}")
        values[name] = statement.value.value
    if set(values) != allowed:
        fail(f"build identity fields do not match in {archive_name}")
    return values


def verify_build_identity(source: bytes, *, archive_name: str, commit: str) -> None:
    actual = parse_build_identity(source, archive_name=archive_name)
    expected = {
        "SOURCE_VERSION": EXPECTED_VERSION,
        "SOURCE_REPOSITORY": EXPECTED_REPOSITORY,
        "SOURCE_COMMIT": commit,
    }
    if actual != expected:
        fail(f"build identity mismatch in {archive_name}: {actual!r}")


def expected_identity(commit: str) -> bytes:
    return (
        '"""Generated release identity; do not commit this file."""\n\n'
        f"SOURCE_VERSION = {EXPECTED_VERSION!r}\n"
        f"SOURCE_REPOSITORY = {EXPECTED_REPOSITORY!r}\n"
        f"SOURCE_COMMIT = {commit!r}\n"
    ).encode("utf-8")


def verify_identity_bytes(source: bytes, *, archive_name: str, commit: str) -> None:
    verify_build_identity(source, archive_name=archive_name, commit=commit)
    if source != expected_identity(commit):
        fail(f"non-canonical build identity bytes in {archive_name}")


def single_header(message: object, name: str, expected: str) -> None:
    values = message.get_all(name, [])  # type: ignore[attr-defined]
    if values != [expected]:
        fail(f"metadata {name!r} mismatch: {values!r}")


def verify_metadata(source: bytes, *, archive_name: str) -> None:
    try:
        message = BytesParser(policy=policy.default).parsebytes(source)
    except Exception as exc:  # email defects vary by Python maintenance release.
        fail(f"cannot parse metadata in {archive_name}: {exc}")
    if message.defects:
        fail(f"metadata defects in {archive_name}: {message.defects!r}")
    for name, expected in (
        ("Metadata-Version", "2.4"),
        ("Name", "schemen-gate"),
        ("Version", EXPECTED_VERSION),
        (
            "Summary",
            "AI PKI primitives for identity-bound Gates, lockboxes, capabilities, and Cargo.",
        ),
        ("Author", "Sekos AI"),
        ("License-Expression", "Apache-2.0"),
        ("Requires-Python", ">=3.10"),
        ("Description-Content-Type", "text/markdown"),
    ):
        single_header(message, name, expected)
    if set(message.get_all("Project-URL", [])) != {
        f"Repository, {EXPECTED_REPOSITORY}",
        f"Issues, {EXPECTED_REPOSITORY}/issues",
        f"Changelog, {EXPECTED_REPOSITORY}/blob/main/CHANGELOG.md",
    }:
        fail(f"metadata project URLs mismatch in {archive_name}")
    requires = message.get_all("Requires-Dist", [])
    if len(requires) != len(set(requires)) or set(requires) != EXPECTED_REQUIRES_DIST:
        fail(f"metadata dependencies mismatch in {archive_name}: {requires!r}")
    extras = message.get_all("Provides-Extra", [])
    if len(extras) != len(set(extras)) or set(extras) != EXPECTED_EXTRAS:
        fail(f"metadata extras mismatch in {archive_name}: {extras!r}")
    if message.get_all("License-File", []) != ["LICENSE", "NOTICE"]:
        fail(f"metadata license files mismatch in {archive_name}")
    if message.get_all("Dynamic", []) != ["license-file"]:
        fail(f"metadata dynamic fields mismatch in {archive_name}")
    body = source.split(b"\n\n", 1)
    if len(body) != 2 or body[1] != git_bytes("PYPI.md"):
        fail(f"metadata long description is not the reviewed PYPI.md in {archive_name}")


def expected_package_files(commit: str) -> dict[str, bytes]:
    files = {
        path.removeprefix("src/"): git_bytes(path)
        for path in tracked_paths()
        if path.startswith("src/schemen_gate/")
    }
    files["schemen_gate/_build_identity.py"] = expected_identity(commit)
    return files


def verify_wheel_record(record: bytes, member_facts: dict[str, tuple[bytes, int]]) -> None:
    record_name = f"{DIST_INFO}/RECORD"
    try:
        rows = list(csv.reader(io.StringIO(record.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        fail(f"invalid wheel RECORD: {exc}")
    if len(rows) != len(member_facts):
        fail("wheel RECORD row count does not match archive members")
    seen: set[str] = set()
    for row in rows:
        if len(row) != 3:
            fail(f"invalid wheel RECORD row: {row!r}")
        name, encoded_digest, size = row
        if name in seen or name not in member_facts:
            fail(f"unexpected or duplicate wheel RECORD path: {name!r}")
        seen.add(name)
        if name == record_name:
            if encoded_digest or size:
                fail("wheel RECORD must not hash itself")
            continue
        raw_digest, member_size = member_facts[name]
        digest = base64.urlsafe_b64encode(raw_digest).rstrip(b"=")
        if encoded_digest != f"sha256={digest.decode('ascii')}" or size != str(member_size):
            fail(f"wheel RECORD digest or size mismatch for {name!r}")
    if seen != set(member_facts):
        fail("wheel RECORD does not cover every archive member")


def verify_wheel(path: Path, *, commit: str) -> int:
    archive_size(path, label="wheel")
    package = expected_package_files(commit)
    identity_name = "schemen_gate/_build_identity.py"
    metadata_name = f"{DIST_INFO}/METADATA"
    record_name = f"{DIST_INFO}/RECORD"
    expected_wheel = (
        b"Wheel-Version: 1.0\n"
        b"Generator: setuptools (84.0.0)\n"
        b"Root-Is-Purelib: true\n"
        b"Tag: py3-none-any\n\n"
    )
    exact_files = {
        **package,
        f"{DIST_INFO}/licenses/LICENSE": git_bytes("LICENSE"),
        f"{DIST_INFO}/licenses/NOTICE": git_bytes("NOTICE"),
        f"{DIST_INFO}/WHEEL": expected_wheel,
        f"{DIST_INFO}/top_level.txt": b"schemen_gate\n",
    }
    expected_names = set(exact_files) | {metadata_name, record_name}
    retained: dict[str, bytes] = {}
    facts: dict[str, tuple[bytes, int]] = {}
    total_size = 0
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) != len(expected_names):
            fail(f"wheel member count mismatch: expected {len(expected_names)}, found {len(infos)}")
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            fail("wheel contains duplicate archive paths")
        for info in infos:
            validate_archive_path(info.filename, label="wheel path")
            if info.is_dir():
                fail(f"wheel contains unexpected directory entry: {info.filename!r}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                fail(f"wheel contains a symlink: {info.filename!r}")
            if info.flag_bits & 0x1:
                fail(f"wheel contains an encrypted member: {info.filename!r}")
            if info.compress_type != zipfile.ZIP_DEFLATED:
                fail(f"wheel uses an unexpected compression method: {info.filename!r}")
        require_exact_members(actual=set(names), expected=expected_names, label="wheel")
        for info in infos:
            validate_compressed_member(
                declared_size=info.file_size,
                compressed_size=info.compress_size,
                label=f"wheel member {info.filename!r}",
            )
            total_size += info.file_size
            if total_size > MAX_TOTAL_MEMBER_BYTES:
                fail("wheel aggregate payload exceeds the release bound")
            expected = exact_files.get(info.filename)
            if expected is not None and info.file_size != len(expected):
                fail(f"wheel member size differs from reviewed bytes: {info.filename}")
            with archive.open(info, "r") as source:
                payload = read_bounded(
                    source,
                    declared_size=info.file_size,
                    label=f"wheel member {info.filename!r}",
                )
            facts[info.filename] = member_fact(payload)
            if expected is not None and payload != expected:
                fail(f"wheel bytes differ from the reviewed contract: {info.filename}")
            if info.filename in {identity_name, metadata_name, record_name}:
                retained[info.filename] = payload

    verify_identity_bytes(retained[identity_name], archive_name=identity_name, commit=commit)
    verify_metadata(retained[metadata_name], archive_name="wheel METADATA")
    verify_wheel_record(retained[record_name], facts)
    return len(facts)


def include_in_sdist(path: str) -> bool:
    if path in {
        "CHANGELOG.md",
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "MANIFEST.in",
        "NOTICE",
        "PYPI.md",
        "README.md",
        "RELEASE_MANIFEST.sha256",
        "ROADMAP.md",
        "SECURITY.md",
        "pyproject.toml",
        "release-contract.json",
    }:
        return True
    rules = (
        ("docs/", (".md",)),
        ("examples/", (".md", ".py")),
        ("requirements/", (".lock",)),
        ("scripts/", (".py", ".sh")),
        ("src/schemen_gate/", (".py", "/py.typed")),
        ("tests/", (".py",)),
    )
    return any(path.startswith(prefix) and path.endswith(suffixes) for prefix, suffixes in rules)


def expected_sdist_sources(commit: str) -> dict[str, bytes]:
    files = {path: git_bytes(path) for path in tracked_paths() if include_in_sdist(path)}
    files["src/schemen_gate/_build_identity.py"] = expected_identity(commit)
    return files


def expected_directories(files: set[str]) -> set[str]:
    directories = {""}
    for name in files:
        parent = PurePosixPath(name).parent
        while parent.as_posix() != ".":
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def verify_sdist(path: Path, *, commit: str) -> int:
    compressed_size = archive_size(path, label="sdist")
    sources = expected_sdist_sources(commit)
    expected_files = set(sources) | GENERATED_SDIST_FILES
    expected_dirs = expected_directories(expected_files)
    expected_member_count = len(expected_files) + len(expected_dirs)
    files: set[str] = set()
    directories: set[str] = set()
    generated: dict[str, bytes] = {}
    identity_name = "src/schemen_gate/_build_identity.py"
    identity_payload: bytes | None = None
    seen_archive_paths: set[str] = set()
    member_count = 0
    total_size = 0
    expanded = decompress_sdist(path, compressed_size=compressed_size)
    try:
        with tarfile.open(fileobj=expanded, mode="r:") as archive:
            for member in archive:
                member_count += 1
                if member_count > expected_member_count:
                    fail(
                        "sdist member count exceeds the exact release allowlist: "
                        f"maximum {expected_member_count}"
                    )
                if member.name in seen_archive_paths:
                    fail(f"sdist contains duplicate archive path: {member.name!r}")
                seen_archive_paths.add(member.name)
                validate_archive_path(member.name, label="sdist path", allow_root=True)
                prefix = f"{SDIST_ROOT}/"
                if member.name == SDIST_ROOT:
                    relative = ""
                elif member.name.startswith(prefix):
                    relative = member.name[len(prefix) :]
                else:
                    fail(f"sdist member is outside its canonical root: {member.name!r}")
                if member.isdir():
                    if relative not in expected_dirs:
                        fail(f"unexpected sdist directory: {relative!r}")
                    directories.add(relative)
                    continue
                if not member.isfile() or member.issparse():
                    fail(f"sdist contains a link or special file: {member.name!r}")
                if relative not in expected_files:
                    fail(f"unexpected sdist file: {relative!r}")
                expected = sources.get(relative)
                if expected is not None and member.size != len(expected):
                    fail(f"sdist member size differs from reviewed bytes: {relative}")
                validate_compressed_member(
                    declared_size=member.size,
                    compressed_size=member.size,
                    label=f"sdist member {relative!r}",
                )
                total_size += member.size
                if total_size > MAX_TOTAL_MEMBER_BYTES:
                    fail("sdist aggregate payload exceeds the release bound")
                extracted = archive.extractfile(member)
                if extracted is None:
                    fail(f"cannot read sdist member: {member.name!r}")
                payload = read_bounded(
                    extracted,
                    declared_size=member.size,
                    label=f"sdist member {relative!r}",
                )
                files.add(relative)
                if expected is not None:
                    if payload != expected:
                        fail(f"sdist source bytes differ from reviewed commit: {relative}")
                    if relative == identity_name:
                        identity_payload = payload
                else:
                    generated[relative] = payload
    finally:
        expanded.close()

    require_exact_members(actual=files, expected=expected_files, label="sdist")
    if directories != expected_dirs:
        fail(
            "sdist directory allowlist mismatch; "
            f"missing={sorted(expected_dirs - directories)!r}, "
            f"unexpected={sorted(directories - expected_dirs)!r}"
        )
    if member_count != expected_member_count:
        fail(f"sdist member count mismatch: expected {expected_member_count}, found {member_count}")
    if identity_payload is None:
        fail("sdist build identity is missing")
    verify_identity_bytes(identity_payload, archive_name=identity_name, commit=commit)
    verify_metadata(generated["PKG-INFO"], archive_name="sdist PKG-INFO")
    if generated["src/schemen_gate.egg-info/PKG-INFO"] != generated["PKG-INFO"]:
        fail("sdist PKG-INFO copies differ")
    if generated["setup.cfg"] != b"[egg_info]\ntag_build = \ntag_date = 0\n\n":
        fail("sdist setup.cfg mismatch")
    if generated["src/schemen_gate.egg-info/dependency_links.txt"] != b"\n":
        fail("sdist dependency_links.txt mismatch")
    if generated["src/schemen_gate.egg-info/top_level.txt"] != b"schemen_gate\n":
        fail("sdist top_level.txt mismatch")
    source_index = generated["src/schemen_gate.egg-info/SOURCES.txt"].decode("utf-8").splitlines()
    expected_index = set(sources) | {
        "src/schemen_gate.egg-info/PKG-INFO",
        "src/schemen_gate.egg-info/SOURCES.txt",
        "src/schemen_gate.egg-info/dependency_links.txt",
        "src/schemen_gate.egg-info/requires.txt",
        "src/schemen_gate.egg-info/top_level.txt",
    }
    if len(source_index) != len(set(source_index)) or set(source_index) != expected_index:
        fail("sdist SOURCES.txt does not exactly index the reviewed payload")
    return member_count


def main() -> int:
    commit = expected_commit()
    wheels = sorted(DIST.glob("schemen_gate-*.whl"))
    sdists = sorted(DIST.glob("schemen_gate-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        fail(f"expected one wheel and one sdist, found {wheels!r} and {sdists!r}")
    wheel_count = verify_wheel(wheels[0], commit=commit)
    sdist_count = verify_sdist(sdists[0], commit=commit)
    print(
        f"distribution verification passed against {commit}: "
        f"{wheel_count} wheel entries, {sdist_count} sdist entries"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
