"""Credential and publication-binary coverage for the release hygiene gate."""

from __future__ import annotations

import hashlib
import re
import struct
import subprocess
import zlib
from pathlib import Path

import pytest
from pypdf import PdfWriter

from scripts import public_hygiene

ROOT = Path(__file__).resolve().parents[1]


def _approved_readme_section() -> bytes:
    data = (ROOT / "README.md").read_bytes()
    return next(
        section.group()
        for section in re.finditer(rb"(?ms)^## [^\n]+\n.*?(?=^## |\Z)", data)
        if hashlib.sha256(section.group()).hexdigest()
        == public_hygiene.APPROVED_README_SECTION_SHA256
    )


def test_reviewed_product_prose_is_allowed_only_in_root_readme() -> None:
    section = _approved_readme_section()
    assert not tuple(public_hygiene._pattern_findings(section, location="README.md"))
    for location in ("docs/README.md", "docs/guide.md", "paper.pdf:page-1"):
        assert tuple(public_hygiene._pattern_findings(section, location=location))


def test_changed_product_prose_requires_new_review() -> None:
    section = _approved_readme_section().replace(b"serving product", b"private source")
    assert tuple(public_hygiene._pattern_findings(section, location="README.md"))


def test_unapproved_product_prose_outside_section_is_still_detected() -> None:
    section = _approved_readme_section()
    data = section + b"## Extra\n" + b"Run" + b"time\n"
    line = section.count(b"\n") + 2
    assert tuple(public_hygiene._pattern_findings(data, location="README.md")) == (
        f"README.md:{line}: product-styled execution term",
    )


def test_duplicate_approved_section_does_not_expand_allowance() -> None:
    section = _approved_readme_section()
    assert tuple(public_hygiene._pattern_findings(section * 2, location="README.md"))


def test_even_approved_prose_cannot_hide_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    section = _approved_readme_section() + b"ghp_" + b"A" * 32 + b"\n"
    monkeypatch.setattr(
        public_hygiene, "APPROVED_README_SECTION_SHA256", hashlib.sha256(section).hexdigest()
    )
    findings = tuple(public_hygiene._pattern_findings(section, location="README.md"))
    line = section.count(b"\n")
    assert findings == (f"README.md:{line}: GitHub token",)


@pytest.mark.parametrize(
    ("label", "secret"),
    [
        ("Modal credential", b"ak-" + b"A" * 32),
        ("Modal credential", b"as-" + b"B" * 32),
        ("Modal credential", b"wk-" + b"C" * 32),
        ("Modal credential", b"ws-" + b"D" * 32),
        ("PyPI token", b"pypi-" + b"E" * 32),
        ("Hugging Face token", b"hf_" + b"F" * 32),
        ("Google API key", b"AIza" + b"G" * 35),
        ("GitLab token", b"glpat-" + b"H" * 32),
        ("private companion package", b"sche" + b"men\nrun" + b"time"),
        ("private product offering", b"gov" + b"erned\n  run" + b"time"),
    ],
)
def test_service_credential_patterns_are_detected(label: str, secret: bytes) -> None:
    findings = tuple(public_hygiene._pattern_findings(secret, location="fixture"))

    assert findings == (f"fixture:1: {label}",)


def test_pdf_metadata_is_extracted_and_scanned(tmp_path: Path) -> None:
    path = tmp_path / "credential.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_metadata({"/Producer": "ak-" + "A" * 32})
    with path.open("wb") as handle:
        writer.write(handle)

    payloads = dict(public_hygiene._pdf_payloads(path))
    findings = tuple(
        public_hygiene._pattern_findings(payloads["metadata"], location="pdf:metadata")
    )

    assert findings == ("pdf:metadata:1: Modal credential",)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def test_png_text_metadata_is_extracted_and_scanned(tmp_path: Path) -> None:
    path = tmp_path / "credential.png"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    image = zlib.compress(b"\x00\x00")
    secret = b"ak-" + b"A" * 32
    path.write_bytes(
        public_hygiene.PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"tEXt", b"Comment\0" + secret)
        + _png_chunk(b"IDAT", image)
        + _png_chunk(b"IEND", b"")
    )

    payloads = dict(public_hygiene._png_payloads(path))
    findings = tuple(public_hygiene._pattern_findings(payloads["text-1"], location="png:text-1"))

    assert findings == ("png:text-1:1: Modal credential",)


def test_every_tracked_binary_has_a_bounded_extractor() -> None:
    names = subprocess.check_output(("git", "ls-files", "-z"), cwd=ROOT).decode().split("\0")
    binary_paths = [
        ROOT / name
        for name in names
        if name and (ROOT / name).is_file() and b"\0" in (ROOT / name).read_bytes()
    ]

    assert binary_paths
    assert all(path.suffix.lower() in {".pdf", ".png"} for path in binary_paths)
    for path in binary_paths:
        if path.suffix.lower() == ".pdf":
            payloads = public_hygiene._pdf_payloads(path)
            assert any(source.startswith("page-") for source, _ in payloads)
        else:
            payloads = public_hygiene._png_payloads(path)
            assert dict(payloads)["raw"].startswith(public_hygiene.PNG_SIGNATURE)
