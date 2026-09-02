"""Credential and publication-binary coverage for the release hygiene gate."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pypdf import PdfWriter

from scripts import public_hygiene

ROOT = Path(__file__).resolve().parents[1]


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


def test_every_tracked_binary_is_an_extracted_pdf() -> None:
    names = subprocess.check_output(("git", "ls-files", "-z"), cwd=ROOT).decode().split("\0")
    binary_paths = [
        ROOT / name
        for name in names
        if name and (ROOT / name).is_file() and b"\0" in (ROOT / name).read_bytes()
    ]

    assert binary_paths
    assert all(path.suffix.lower() == ".pdf" for path in binary_paths)
    for path in binary_paths:
        payloads = public_hygiene._pdf_payloads(path)
        assert any(source.startswith("page-") for source, _ in payloads)
