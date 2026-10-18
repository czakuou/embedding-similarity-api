"""Worker slice: an arq job (`jobs.py`), greedy micro-batching in front of the encoder
(`batcher.py`) and the arq wiring (`app.py`, `entrypoint.py`).
"""

from __future__ import annotations
