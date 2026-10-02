"""Roadmap feature 12 (offline mode and audit): signed scenario packs.

A "pack" is a scripted scenario's JSON plus an HMAC-SHA256 signature over
its exact bytes, so a scenario file distributed to an air-gapped site (no
network, no central server to check against) can still be verified as
unmodified since it was signed — tamper-evidence without connectivity.

The signing key is local-only (a file next to the scripted scenarios,
generated on first use) — this is a trainer, not a PKI system; the goal is
detecting accidental or malicious edits to a distributed scenario pack, not
defending against a well-resourced adversary with access to the signing
host.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path

KEY_PATH = Path(__file__).parent / "scripted" / ".pack_signing_key"


def _get_or_create_key() -> bytes:
    if KEY_PATH.exists():
        return bytes.fromhex(KEY_PATH.read_text().strip())
    key = secrets.token_bytes(32)
    KEY_PATH.write_text(key.hex())
    return key


def sign_pack(scenario: dict) -> dict:
    """Returns {"scenario": scenario, "signature": hex_hmac} — the
    signature covers the exact canonical JSON bytes of `scenario`."""
    key = _get_or_create_key()
    payload = json.dumps(scenario, sort_keys=True).encode()
    signature = hmac.new(key, payload, hashlib.sha256).hexdigest()
    return {"scenario": scenario, "signature": signature}


def verify_pack(pack: dict) -> bool:
    """Recomputes the HMAC over `pack["scenario"]` and compares it
    constant-time against `pack["signature"]`. False on any mismatch or
    malformed pack — never raises, since a broken/tampered pack should
    just fail verification, not crash the caller."""
    try:
        key = _get_or_create_key()
        payload = json.dumps(pack["scenario"], sort_keys=True).encode()
        expected = hmac.new(key, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, pack["signature"])
    except (KeyError, TypeError):
        return False
