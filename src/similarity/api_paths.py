"""The one place that knows the API's version prefix and resource path.

submit.py and results.py both build /v1/similarity-analyses/{id} URLs (the
POST response's Location, and links.self everywhere); bumping the API
version should mean touching this file and nothing else.
"""

from __future__ import annotations

API_V1_PREFIX = "/v1"
SIMILARITY_ANALYSES_PATH = f"{API_V1_PREFIX}/similarity-analyses"
SIMILARITY_ANALYSIS_ITEM_ROUTE = f"{SIMILARITY_ANALYSES_PATH}/{{analysis_id}}"


def analysis_url(analysis_id: str) -> str:
    return f"{SIMILARITY_ANALYSES_PATH}/{analysis_id}"
