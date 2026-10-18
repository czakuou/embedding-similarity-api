from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette import status

from similarity.config import WorkerSettings
from similarity.encoder import SentenceTransformerEncoder

from .conftest import process_queue, submit_valid_document

pytestmark = pytest.mark.behavior


async def test_given_done_analysis_when_deleted_then_204_and_then_404(
    app_client: TestClient,
    worker_settings: WorkerSettings,
    real_encoder: SentenceTransformerEncoder,
) -> None:
    response = submit_valid_document(app_client, "client-a")
    analysis_id = response.json()["id"]
    await process_queue(worker_settings, real_encoder)
    done = app_client.get(
        f"/v1/similarity-analyses/{analysis_id}", headers={"X-Client-Id": "client-a"}
    )
    assert done.json()["status"] == "done"

    delete = app_client.delete(
        f"/v1/similarity-analyses/{analysis_id}", headers={"X-Client-Id": "client-a"}
    )
    assert delete.status_code == status.HTTP_204_NO_CONTENT, delete.text

    poll_after_delete = app_client.get(
        f"/v1/similarity-analyses/{analysis_id}", headers={"X-Client-Id": "client-a"}
    )
    assert poll_after_delete.status_code == status.HTTP_404_NOT_FOUND, poll_after_delete.text


def test_given_in_progress_analysis_when_deleted_then_409(app_client: TestClient) -> None:
    response = submit_valid_document(app_client, "client-a")
    analysis_id = response.json()["id"]

    delete = app_client.delete(
        f"/v1/similarity-analyses/{analysis_id}", headers={"X-Client-Id": "client-a"}
    )
    assert delete.status_code == status.HTTP_409_CONFLICT, delete.text

    still_there = app_client.get(
        f"/v1/similarity-analyses/{analysis_id}", headers={"X-Client-Id": "client-a"}
    )
    assert still_there.json()["status"] == "queued"
