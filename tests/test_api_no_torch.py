from __future__ import annotations

import subprocess
import sys


def test_given_api_module_when_imported_then_torch_is_not_loaded() -> None:
    """The lightweight API image must never pull in torch (SPEC §3.5)."""
    result = subprocess.run(
        [sys.executable, "-c", "import similarity.api; import sys; print('torch' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
