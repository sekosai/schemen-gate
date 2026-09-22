"""Malformed delegated requests must return bounded, controlled denials."""

from __future__ import annotations

import pytest
from credential_broker.delegation._profile import Denied, loads
from credential_broker.delegation.api import callback_application
from credential_broker.delegation.demo import BoundHarness
from fastapi.testclient import TestClient

DEEP_JSON = b'{"statement":' + b"[" * 10000 + b"0" + b"]" * 10000 + b"}"


def test_delegation_json_recursion_is_a_controlled_denial():
    with pytest.raises(Denied) as error:
        loads(DEEP_JSON)
    assert error.value.code == "invalid_request"


@pytest.mark.parametrize("endpoint", ["/requests", "/reviews", "/approvals"])
def test_deep_delegation_json_denied_before_execution(tmp_path, endpoint):
    harness = BoundHarness(tmp_path)
    response = harness.http.post(
        endpoint, content=DEEP_JSON, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 403
    assert response.json() == {"error": "invalid_request"}
    assert response.headers["Cache-Control"] == "no-store"
    assert not harness.executor.calls


def test_deep_callback_json_is_a_controlled_denial(tmp_path):
    harness = BoundHarness(tmp_path)
    client = TestClient(callback_application(harness.inbox))
    response = client.post(
        harness.inbox.url, content=DEEP_JSON, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 403
    assert response.json() == {"error": "invalid_callback"}
    assert not harness.executor.calls
