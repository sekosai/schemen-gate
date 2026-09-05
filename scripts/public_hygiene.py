#!/usr/bin/env python3
"""Fail closed on credential signatures and local-machine residue."""

from __future__ import annotations

import hashlib
import re
import struct
import subprocess  # nosec B404
import zlib
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 5_000_000
MAX_PDF_PAGES = 500
MAX_PDF_EXTRACTED_BYTES = 20_000_000
MAX_PNG_PIXELS = 100_000_000
MAX_PNG_TEXT_BYTES = 1_000_000
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Only the reviewed README product-boundary section may name the companion.
# Keep its bytes pinned: editing that prose requires an explicit policy review.
APPROVED_README_SECTION_SHA256 = "8ef4593d4d5297ed74c1fca5b424087f8a161c075d810d9a0f999eb9dc75451f"
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


def _bounded_zlib(data: bytes, *, location: str) -> bytes:
    """Decompress one PNG text payload without permitting a decompression bomb."""

    decompressor = zlib.decompressobj()
    output = decompressor.decompress(data, MAX_PNG_TEXT_BYTES + 1)
    if len(output) > MAX_PNG_TEXT_BYTES or decompressor.unconsumed_tail:
        raise RuntimeError(f"PNG compressed text exceeds the scan limit: {location}")
    output += decompressor.flush(MAX_PNG_TEXT_BYTES + 1 - len(output))
    if len(output) > MAX_PNG_TEXT_BYTES or not decompressor.eof:
        raise RuntimeError(f"PNG compressed text is invalid or oversized: {location}")
    return output


def _png_payloads(path: Path) -> tuple[tuple[str, bytes], ...]:
    """Validate a bounded PNG and expose raw plus textual metadata for scanning."""

    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise RuntimeError(f"tracked PNG has an invalid signature: {path.name}")

    payloads: list[tuple[str, bytes]] = [("raw", data)]
    offset = len(PNG_SIGNATURE)
    saw_ihdr = False
    saw_idat = False
    saw_iend = False
    text_bytes = 0
    text_index = 0

    while offset < len(data):
        if len(data) - offset < 12:
            raise RuntimeError(f"tracked PNG has a truncated chunk: {path.name}")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(data):
            raise RuntimeError(f"tracked PNG has an oversized chunk: {path.name}")
        if len(chunk_type) != 4 or not all(
            65 <= byte <= 90 or 97 <= byte <= 122 for byte in chunk_type
        ):
            raise RuntimeError(f"tracked PNG has an invalid chunk type: {path.name}")
        if chunk_type[2] & 0x20:
            raise RuntimeError(f"tracked PNG violates the reserved chunk bit: {path.name}")

        chunk_data = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : chunk_end])[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(chunk_data, actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise RuntimeError(f"tracked PNG has a chunk CRC mismatch: {path.name}")
        if not saw_ihdr and chunk_type != b"IHDR":
            raise RuntimeError(f"tracked PNG does not begin with IHDR: {path.name}")

        if chunk_type == b"IHDR":
            if saw_ihdr or length != 13:
                raise RuntimeError(f"tracked PNG has an invalid IHDR: {path.name}")
            width, height = struct.unpack(">II", chunk_data[:8])
            if width == 0 or height == 0 or width * height > MAX_PNG_PIXELS:
                raise RuntimeError(f"tracked PNG dimensions exceed policy: {path.name}")
            if chunk_data[10] != 0 or chunk_data[11] != 0 or chunk_data[12] not in (0, 1):
                raise RuntimeError(f"tracked PNG uses unsupported encoding: {path.name}")
            saw_ihdr = True
        elif chunk_type == b"IDAT":
            saw_idat = True
        elif chunk_type == b"tEXt":
            keyword, separator, text = chunk_data.partition(b"\0")
            if not separator or not keyword:
                raise RuntimeError(f"tracked PNG has malformed tEXt: {path.name}")
            text_index += 1
            text_bytes += len(keyword) + len(text)
            payloads.append((f"text-{text_index}", keyword + b"=" + text))
        elif chunk_type == b"zTXt":
            keyword, separator, remainder = chunk_data.partition(b"\0")
            if not separator or not keyword or not remainder or remainder[0] != 0:
                raise RuntimeError(f"tracked PNG has malformed zTXt: {path.name}")
            text = _bounded_zlib(remainder[1:], location=path.name)
            text_index += 1
            text_bytes += len(keyword) + len(text)
            payloads.append((f"text-{text_index}", keyword + b"=" + text))
        elif chunk_type == b"iTXt":
            keyword, separator, remainder = chunk_data.partition(b"\0")
            if not separator or not keyword or len(remainder) < 2:
                raise RuntimeError(f"tracked PNG has malformed iTXt: {path.name}")
            compression_flag, compression_method = remainder[:2]
            language, separator, remainder = remainder[2:].partition(b"\0")
            translated, separator_two, text = remainder.partition(b"\0")
            if not separator or not separator_two or compression_method != 0:
                raise RuntimeError(f"tracked PNG has malformed iTXt: {path.name}")
            if compression_flag == 1:
                text = _bounded_zlib(text, location=path.name)
            elif compression_flag != 0:
                raise RuntimeError(f"tracked PNG has invalid iTXt compression: {path.name}")
            text_index += 1
            text_bytes += len(keyword) + len(language) + len(translated) + len(text)
            payloads.append((f"text-{text_index}", keyword + b"=" + translated + b"\n" + text))
        elif chunk_type == b"IEND":
            if length != 0 or chunk_end != len(data):
                raise RuntimeError(f"tracked PNG has an invalid IEND: {path.name}")
            saw_iend = True

        if text_bytes > MAX_PNG_TEXT_BYTES:
            raise RuntimeError(f"tracked PNG metadata exceeds the scan limit: {path.name}")
        offset = chunk_end

    if not saw_ihdr or not saw_idat or not saw_iend:
        raise RuntimeError(f"tracked PNG is missing a required chunk: {path.name}")
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
    checked_png = 0
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
        if path.suffix.lower() == ".png":
            try:
                payloads = _png_payloads(path)
            except RuntimeError as exc:
                findings.append(f"{relative}: PNG hygiene scan failed: {exc}")
                continue
            checked_png += 1
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
        "Public hygiene: "
        f"{checked_text} tracked text files, {checked_pdf} extracted PDFs, "
        f"and {checked_png} validated PNGs passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
