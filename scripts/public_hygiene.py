#!/usr/bin/env python3
"""Fail closed on credential signatures and local-machine residue."""

from __future__ import annotations

import hashlib
import re
import subprocess  # nosec B404
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 5_000_000
MAX_PDF_PAGES = 500
MAX_PDF_EXTRACTED_BYTES = 20_000_000

# Only the reviewed README product-boundary section may name the companion.
# Keep its bytes pinned: editing that prose requires an explicit policy review.
APPROVED_README_SECTION_SHA256 = "c6b4ed05c285f219ea1b1710856217276fe6ab5c120504bdcad04979c6c0ff90"
README_PRODUCT_LABELS = frozenset({"private companion package", "product-styled execution term"})

FORBIDDEN_BASENAMES = frozenset(
    {
        ".env",
        ".npmrc",
        ".pypirc",
        "credentials.json",
        "id_ed25519",
        "id_rsa",
    }
)
FORBIDDEN_SUFFIXES = frozenset({".key", ".p12", ".pfx", ".whl"})
FORBIDDEN_PATH_FRAGMENTS = (
    "GOVERNED_" + "RUNTIME",
    "binary-activation-" + "runtime",
    "runtime_" + "authorization",
)

PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "private-key block",
        re.compile(
            rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----\s+"
            rb"[A-Za-z0-9+/=\r\n]{32,}"
        ),
    ),
    ("AWS access key", re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "GitHub token",
        re.compile(rb"\b(?:gh[opurs]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    (
        "OpenAI-style token",
        re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    ),
    (
        "Modal credential",
        re.compile(rb"\b(?:ak|as|wk|ws)-[A-Za-z0-9_-]{20,}\b"),
    ),
    ("PyPI token", re.compile(rb"\bpypi-[A-Za-z0-9_-]{20,}\b")),
    ("Hugging Face token", re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b")),
    ("Google API key", re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("GitLab token", re.compile(rb"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    ("Slack token", re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    (
        "JWT",
        re.compile(
            rb"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."
            rb"[A-Za-z0-9_-]{10,}\b"
        ),
    ),
    (
        "credential in URL",
        re.compile(rb"https?://[^/\s:@]+:[^/\s@]+@", re.IGNORECASE),
    ),
    (
        "absolute macOS user path",
        re.compile(rb"/Users/[A-Za-z0-9._-]+/"),
    ),
    (
        "absolute Linux user path",
        re.compile(rb"/home/[A-Za-z0-9._-]+/"),
    ),
    (
        "absolute temporary checkout path",
        re.compile(b"/pri" + rb"vate/(?:tmp|var)/"),
    ),
    ("local file URL", re.compile(rb"file:///(?:Users|home|private)/")),
    ("assistant worktree residue", re.compile(rb"\." + rb"codex-worktrees/")),
    (
        "private patent identifier",
        re.compile(rb"(?:[0-9]{2}/[0-9]{3},[0-9]{3}|[A-Z]{4}\.[0-9]{3}PR)"),
    ),
    (
        "private companion package",
        re.compile(rb"\b" + b"schemen" + rb"[\s_-]+" + b"runtime" + rb"\b", re.IGNORECASE),
    ),
    (
        "private product offering",
        re.compile(b"governed" + rb"[\s-]+" + b"runtime", re.IGNORECASE),
    ),
    (
        "product-styled execution term",
        re.compile(rb"\b" + b"Run" + b"time" + rb"(?:\b|Execution)"),
    ),
)


def _product_scan_data(data: bytes, *, location: str) -> bytes:
    """Mask exact approved prose for product-name checks, preserving offsets."""
    if location != "README.md":
        return data
    for section in re.finditer(rb"(?ms)^## [^\n]+\n.*?(?=^## |\Z)", data):
        if hashlib.sha256(section.group()).hexdigest() == APPROVED_README_SECTION_SHA256:
            masked = re.sub(rb"[^\n]", b" ", section.group())
            return data[: section.start()] + masked + data[section.end() :]
    return data


def _pattern_findings(data: bytes, *, location: str) -> Iterable[str]:
    product_data = _product_scan_data(data, location=location)
    for label, pattern in PATTERNS:
        # Credentials, machine residue, and all other checks see the full text.
        match = pattern.search(product_data if label in README_PRODUCT_LABELS else data)
        if match is None:
            continue
        line = data[: match.start()].count(b"\n") + 1
        yield f"{location}:{line}: {label}"


def _pdf_payloads(path: Path) -> tuple[tuple[str, bytes], ...]:
    """Extract bounded PDF text and metadata for the credential scan."""

    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:
        raise RuntimeError("PDF scanning requires the release-check dependency pypdf") from exc

    try:
        reader = PdfReader(path, strict=True)
    except (OSError, PdfReadError, ValueError) as exc:
        raise RuntimeError(f"cannot parse tracked PDF {path.name}: {exc}") from exc
    if reader.is_encrypted:
        raise RuntimeError(f"tracked PDF must not be encrypted: {path.name}")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise RuntimeError(f"tracked PDF exceeds the {MAX_PDF_PAGES}-page scan limit: {path.name}")

    payloads: list[tuple[str, bytes]] = [("raw", path.read_bytes())]
    metadata = reader.metadata
    if metadata:
        encoded_metadata = "\n".join(
            f"{key}={value}" for key, value in sorted(metadata.items())
        ).encode("utf-8", errors="replace")
        payloads.append(("metadata", encoded_metadata))

    extracted = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except (KeyError, TypeError, ValueError, PdfReadError) as exc:
            raise RuntimeError(
                f"cannot extract tracked PDF {path.name} page {page_number}: {exc}"
            ) from exc
        encoded = text.encode("utf-8", errors="replace")
        extracted += len(encoded)
        if extracted > MAX_PDF_EXTRACTED_BYTES:
            raise RuntimeError(
                f"tracked PDF extracted text exceeds the aggregate scan limit: {path.name}"
            )
        payloads.append((f"page-{page_number}", encoded))
    return tuple(payloads)


def tracked_paths() -> tuple[Path, ...]:
    output = subprocess.check_output(  # nosec B603
        ("git", "ls-files", "-z"), cwd=ROOT
    ).decode()
    return tuple(ROOT / name for name in output.split("\0") if name)


def main() -> int:
    findings: list[str] = []
    checked_text = 0
    checked_pdf = 0
    for path in tracked_paths():
        relative = path.relative_to(ROOT)
        if any(fragment in relative.as_posix() for fragment in FORBIDDEN_PATH_FRAGMENTS):
            findings.append(f"{relative}: prohibited private-product path")
            continue
        if path.name in FORBIDDEN_BASENAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(f"{relative}: prohibited artifact filename")
            continue
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_TEXT_BYTES:
            findings.append(
                f"{relative}: tracked artifact exceeds the {MAX_TEXT_BYTES}-byte scan limit"
            )
            continue
        if path.suffix.lower() == ".pdf":
            try:
                payloads = _pdf_payloads(path)
            except RuntimeError as exc:
                findings.append(f"{relative}: PDF hygiene scan failed: {exc}")
                continue
            checked_pdf += 1
            for source, data in payloads:
                findings.extend(_pattern_findings(data, location=f"{relative}:{source}"))
            continue
        data = path.read_bytes()
        if b"\0" in data:
            findings.append(f"{relative}: unsupported tracked binary artifact")
            continue
        checked_text += 1
        findings.extend(_pattern_findings(data, location=str(relative)))
    if findings:
        raise SystemExit("public hygiene check failed:\n" + "\n".join(findings))
    print(
        f"Public hygiene: {checked_text} tracked text files and {checked_pdf} extracted PDFs passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
