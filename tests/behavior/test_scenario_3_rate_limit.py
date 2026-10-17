from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette import status

from .conftest import submit_valid_document

pytestmark = pytest.mark.behavior


def test_given_client_over_limit_when_submitting_then_429_and_other_client_unaffected(
    app_client: TestClient,
) -> None:
    # The api_settings fixture sets rate_limit="2/minute" (BEHAVIOR.md testing
    # notes: a long window avoids racing a real 1-second window).
    for _ in range(2):
        response = submit_valid_document(app_client, "client-a")
        assert response.status_code == status.HTTP_202_ACCEPTED, response.text

    third = submit_valid_document(app_client, "client-a")
    assert third.status_code == status.HTTP_429_TOO_MANY_REQUESTS, third.text
    retry_after = int(third.headers["Retry-After"])
    assert 5 <= retry_after <= 10  # base=5 (default), base..2*base
    assert third.json()["code"] == "RATE_LIMITED"

    other_client = submit_valid_document(app_client, "client-b")
    assert other_client.status_code == status.HTTP_202_ACCEPTED, other_client.text
