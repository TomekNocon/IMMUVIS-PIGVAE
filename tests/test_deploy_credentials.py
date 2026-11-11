import importlib.util
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("credentials", ROOT / "scripts/deploy/credentials.py")
credentials = importlib.util.module_from_spec(spec)
spec.loader.exec_module(credentials)


def test_set_env_replaces_in_place_and_appends(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# keep me\nMODAL_KEY=old\nOTHER=1\n")
    credentials.set_env(env, {"MODAL_KEY": "new", "UI_PASSWORD": "pw"}, comment="added")
    assert env.read_text() == "# keep me\nMODAL_KEY=new\nOTHER=1\n\n# added\nUI_PASSWORD=pw\n"
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert credentials.read_env(env) == {"MODAL_KEY": "new", "OTHER": "1", "UI_PASSWORD": "pw"}


def test_read_env_missing_file_and_values_with_equals(tmp_path):
    assert credentials.read_env(tmp_path / "nope") == {}
    env = tmp_path / ".env"
    env.write_text("A=x=y\n  # c=d\n\n")
    assert credentials.read_env(env) == {"A": "x=y"}


def test_find_token_anywhere_in_json():
    data = {"token": {"id": "wk-123", "meta": [{"secret": "ws-456"}]}, "name": "x"}
    assert credentials.find_token(data) == ("wk-123", "ws-456")
    with pytest.raises(SystemExit):
        credentials.find_token({"id": "wk-123"})


@pytest.mark.parametrize(
    ("has_pw", "exists", "force", "new", "expected"),
    [
        (False, False, False, False, (True, True)),  # first setup: generate and create
        (True, False, False, False, (False, True)),  # first setup with your own password
        (True, True, False, False, (False, False)),  # already set up: change nothing
        (True, True, True, False, (False, True)),  # FORCE=1: push the edited .env
        (True, True, False, True, (True, True)),  # NEW_PASSWORD=1
        (False, True, False, True, (True, True)),  # lost .env: NEW_PASSWORD=1 recovers
    ],
)
def test_ui_secret_plan(has_pw, exists, force, new, expected):
    assert credentials.ui_secret_plan(has_pw, exists, force, new) == expected


def test_ui_secret_plan_refuses_mismatch():
    # Secret on Modal but no password in .env: generating one locally would not match Modal.
    with pytest.raises(SystemExit, match="NEW_PASSWORD=1"):
        credentials.ui_secret_plan(False, True, False, False)
    with pytest.raises(SystemExit):
        credentials.ui_secret_plan(False, True, True, False)
