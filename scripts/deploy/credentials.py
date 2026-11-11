"""Credentials for the Modal deployment, kept in the repo-root ``.env`` (gitignored) and in Modal.

Subcommands (run from the repo root; ``make modal-secret`` / ``make modal-token`` call them)::

    uv run python scripts/deploy/credentials.py ui-secret [--force | --new-password]
    uv run python scripts/deploy/credentials.py proxy-token [--replace]

``ui-secret``: the UI login lives in ``.env`` (``UI_PASSWORD``, ``UI_ORIGINS``) and in the Modal
Secret ``pigvae-imc-ui`` (plus a signing key). ``.env`` is the source of truth; nothing that
exists is changed unless asked:

- the Secret does not exist yet: create it from ``.env`` (a random ``UI_PASSWORD`` is generated
  and saved there if missing; ``UI_ORIGINS`` defaults to ``http://localhost:5173``);
- the Secret exists: do nothing;
- ``--force``: push the current ``.env`` values (after editing the password or the origins);
- ``--new-password``: generate a new random password, save it to ``.env``, push it.

Every push uses a fresh signing key, which signs everyone out. Redeploy afterwards
(``make modal-deploy``) so the gateway reads the new values.

``proxy-token``: creates a Modal proxy-auth token (for ``scripts/deploy/query_modal.py``) and
stores it as ``MODAL_KEY`` / ``MODAL_SECRET`` in ``.env``.

Secret values are never printed.
"""

import argparse
import json
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"
UI_SECRET = "pigvae-imc-ui"  # noqa: S105 - the Secret's name, not a password
DEFAULT_ORIGINS = "http://localhost:5173"


def read_env(path: Path) -> dict[str, str]:
    """``KEY=VALUE`` lines of a dotenv file (comments and blank lines skipped)."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def set_env(path: Path, values: dict[str, str], comment: str | None = None) -> None:
    """Set keys in a dotenv file: replace existing lines in place, append new ones. Mode 600."""
    lines = path.read_text().splitlines() if path.exists() else []
    todo = dict(values)
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if not line.lstrip().startswith("#") and key in todo:
            lines[i] = f"{key}={todo.pop(key)}"
    if todo:
        if lines and lines[-1].strip():
            lines.append("")
        if comment:
            lines.append(f"# {comment}")
        lines += [f"{k}={v}" for k, v in todo.items()]
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o600)


def modal(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    """Run the Modal CLI of this environment; its stdout is hidden unless captured."""
    return subprocess.run(  # noqa: S603 - fixed modal CLI arguments
        [sys.executable, "-m", "modal", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
    )


def secret_exists(name: str) -> bool:
    out = modal("secret", "list", "--json", capture=True)
    return any(s.get("name") == name for s in json.loads(out.stdout))


def ui_secret_plan(
    env_has_password: bool, exists: bool, force: bool, new_password: bool
) -> tuple[bool, bool]:
    """(generate a new password?, push to Modal?) for ``ui-secret``."""
    if new_password:
        return True, True
    if not exists:
        return not env_has_password, True
    if not env_has_password:
        # Modal cannot give the password back, so a new one in .env alone would not match.
        raise SystemExit(
            f"The Modal Secret {UI_SECRET} exists but .env has no UI_PASSWORD. "
            "Run `make modal-secret NEW_PASSWORD=1` to set a new password."
        )
    return False, force


def ui_secret(force: bool, new_password: bool) -> None:
    env = read_env(ENV_FILE)
    exists = secret_exists(UI_SECRET)
    generate, push = ui_secret_plan(bool(env.get("UI_PASSWORD")), exists, force, new_password)
    if not push:
        print(
            f"Modal Secret {UI_SECRET} already exists; nothing changed. After editing UI_PASSWORD "
            "or UI_ORIGINS in .env: `make modal-secret FORCE=1`; for a new random password: "
            "`make modal-secret NEW_PASSWORD=1`."
        )
        return
    updates = {}
    if generate:
        updates["UI_PASSWORD"] = secrets.token_urlsafe(18)
    if not env.get("UI_ORIGINS"):
        updates["UI_ORIGINS"] = DEFAULT_ORIGINS
    if updates:
        set_env(
            ENV_FILE,
            updates,
            f"Web UI login, pushed to the Modal Secret {UI_SECRET} by `make modal-secret`. "
            "UI_ORIGINS: allowed browser origins, comma-separated.",
        )
        env.update(updates)

    payload = {
        "UI_PASSWORD": env["UI_PASSWORD"],
        "UI_SIGNING_KEY": secrets.token_hex(32),
        "UI_ORIGINS": env["UI_ORIGINS"],
    }
    with tempfile.TemporaryDirectory() as tmp:  # private directory, removed afterwards
        dotenv = Path(tmp) / "ui.env"
        dotenv.touch(mode=0o600)
        dotenv.write_text("".join(f"{k}={v}\n" for k, v in payload.items()))
        modal("secret", "create", UI_SECRET, "--force", "--from-dotenv", str(dotenv))
    what = "new random password, saved to .env" if generate else "password from .env"
    verb = "updated" if exists else "created"
    print(f"Modal Secret {UI_SECRET} {verb} ({what}; origins: {env['UI_ORIGINS']}).")
    print("Run `make modal-deploy` so the gateway uses it. The password is in .env (UI_PASSWORD).")


def find_token(data: Any) -> tuple[str, str]:
    """The proxy token id (``wk-...``) and secret (``ws-...``) anywhere in the CLI's JSON."""
    found: dict[str, str] = {}

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and x[:3] in ("wk-", "ws-"):
            found.setdefault(x[:3], x)

    walk(data)
    if set(found) != {"wk-", "ws-"}:
        raise SystemExit(
            "could not find the token id and secret in `modal workspace proxy-tokens` output"
        )
    return found["wk-"], found["ws-"]


def proxy_token(replace: bool) -> None:
    if read_env(ENV_FILE).get("MODAL_KEY") and not replace:
        raise SystemExit(".env already has MODAL_KEY; pass --replace to create a new token")
    out = modal(
        "workspace",
        "proxy-tokens",
        "create",
        "--name",
        "pigvae-imc-scripts",
        "--json",
        capture=True,
    )
    key, secret = find_token(json.loads(out.stdout))
    set_env(
        ENV_FILE,
        {"MODAL_KEY": key, "MODAL_SECRET": secret},
        "Modal proxy-auth token for scripts/deploy/query_modal.py (`make modal-token`).",
    )
    print("Proxy token created and saved to .env (MODAL_KEY / MODAL_SECRET).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ui-secret", help="UI password -> .env and the Modal Secret")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="push the .env values to Modal")
    mode.add_argument("--new-password", action="store_true", help="new random password, push it")
    p = sub.add_parser("proxy-token", help="create a proxy-auth token for scripts")
    p.add_argument("--replace", action="store_true", help="create one even if .env has one")
    args = parser.parse_args()
    if args.cmd == "ui-secret":
        ui_secret(args.force, args.new_password)
    else:
        proxy_token(args.replace)


if __name__ == "__main__":
    main()
