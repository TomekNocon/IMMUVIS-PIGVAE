"""The UI's contract files must match the browser API: the OpenAPI file and the mock fixtures."""

import importlib.util
import json
from pathlib import Path

import pytest

from src.deploy.schemas import ErrorResponse, InfoResponse, PredictResponse

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "ui" / "src" / "api" / "fixtures"


def _export_module():
    spec = importlib.util.spec_from_file_location(
        "export_openapi", ROOT / "scripts" / "deploy" / "export_openapi.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_openapi_file_is_up_to_date():
    mod = _export_module()
    assert mod.OUT.read_text() == mod.openapi_json(), (
        "src/deploy/openapi.json is stale: run `uv run python scripts/deploy/export_openapi.py` "
        "and `cd ui && pnpm gen:api`"
    )


@pytest.mark.parametrize(
    ("name", "model"),
    [
        ("info.json", InfoResponse),
        ("predict_adenocarcinoma.json", PredictResponse),
        ("predict_squamous.json", PredictResponse),
        ("error_422.json", ErrorResponse),
    ],
)
def test_fixture_matches_schema(name, model):
    data = json.loads((FIXTURES / name).read_text())
    parsed = model.model_validate(data)
    # No fields beyond the schema: the mock must not show the UI data the real API lacks.
    assert set(data) == set(parsed.model_dump()), set(data) ^ set(parsed.model_dump())


def test_predict_fixtures_are_one_per_class():
    preds = {
        json.loads((FIXTURES / f).read_text())["prediction"]
        for f in ("predict_adenocarcinoma.json", "predict_squamous.json")
    }
    assert preds == set(json.loads((FIXTURES / "info.json").read_text())["classes"])
