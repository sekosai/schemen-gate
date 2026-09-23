"""The application-specific RAR type. Standards do not define Calendar semantics."""

from __future__ import annotations

import json
from typing import Any

from credential_broker.calendar import calendar_request
from credential_broker.json_codec import check_depth
from credential_broker.models import BrokerError, canonical

ISSUER = "https://issuer.example"
TOKEN_URL = ISSUER + "/token"
RESOURCE = "https://broker.example/calendar/execute"
RAR_TYPE = "urn:schemen:authorization:calendar-event-v1"
JWT_TYPE = "urn:ietf:params:oauth:token-type:jwt"
ACCESS_TYPE = "urn:ietf:params:oauth:token-type:access_token"
EXCHANGE = "urn:ietf:params:oauth:grant-type:token-exchange"


class Denied(Exception):
    def __init__(self, code: str = "invalid_grant") -> None:
        self.code = code


def require(condition: bool, code: str = "invalid_grant") -> None:
    if not condition:
        raise Denied(code)


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        require(key not in result, "invalid_request")
        result[key] = value
    return result


def loads(value: str | bytes) -> Any:
    def invalid(_: str) -> None:
        raise Denied("invalid_request")

    try:
        check_depth(value)
        return json.loads(value, object_pairs_hook=_pairs, parse_constant=invalid)
    except (ValueError, UnicodeError, RecursionError):
        raise Denied("invalid_request") from None


def details(value: Any, *, resource: str = RESOURCE) -> list[dict[str, Any]]:
    require(type(value) is list and len(value) == 1, "invalid_authorization_details")
    item = value[0]
    require(type(item) is dict, "invalid_authorization_details")
    require(set(item) == {"type", "actions", "locations", "call"}, "invalid_authorization_details")
    require(
        item["type"] == RAR_TYPE
        and item["actions"] == ["create"]
        and item["locations"] == [resource],
        "invalid_authorization_details",
    )
    try:
        calendar_request(item["call"])
    except (BrokerError, TypeError):
        raise Denied("invalid_authorization_details") from None
    # Freeze JSON by value so the original caller cannot change the approval.
    return loads(canonical(value))  # type: ignore[no-any-return]
