"""Write the browser API's OpenAPI schema to ``src/deploy/openapi.json``.

The UI generates its TypeScript types from this file (``cd ui && pnpm gen:api``), and
``tests/test_deploy_contract.py`` fails when it is out of date. Run after changing
``src/deploy/ui_api.py`` or ``src/deploy/schemas.py``::

    uv run python scripts/deploy/export_openapi.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "src" / "deploy" / "openapi.json"


def openapi_json() -> str:
    from src.deploy.ui_api import create_ui_app

    def unused(*_args):
        raise NotImplementedError

    app = create_ui_app(unused, unused, password="", signing_key=b"x", allowed_origins=[])
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(openapi_json())
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
