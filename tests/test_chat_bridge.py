import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("chat_bridge", ROOT / "scripts/deploy/chat_bridge.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)

ORIGIN = "http://localhost:5173"


def test_build_prompt_has_rules_output_and_conversation():
    msgs = [
        {"role": "user", "content": "why?"},
        {"role": "assistant", "content": "because"},
        {"role": "user", "content": "and crop 0?"},
    ]
    p = cb.build_prompt("RULES", {"prediction": "Adeno"}, msgs, with_system=True)
    assert "<rules>\nRULES" in p and '"prediction": "Adeno"' in p
    assert p.index("User: why?") < p.index("Assistant: because") < p.index("User: and crop 0?")
    assert "<rules>" not in cb.build_prompt("RULES", {}, msgs, with_system=False)


def test_backend_commands_are_locked_down(tmp_path):
    codex = cb.backend_command("codex", tmp_path, "S", None)
    assert codex[:2] == ["codex", "exec"] and codex[-1] == "-"
    assert "read-only" in codex and "--ephemeral" in codex and str(tmp_path / "reply.txt") in codex
    claude = cb.backend_command("claude", tmp_path, "S", "sonnet")
    assert claude[:2] == ["claude", "-p"]
    assert claude[claude.index("--tools") + 1] == ""  # no tools at all
    assert claude[claude.index("--system-prompt") + 1] == "S" and claude[-2:] == [
        "--model",
        "sonnet",
    ]
    with pytest.raises(ValueError):
        cb.backend_command("gpt", tmp_path, "S", None)


class FakeBridge(cb.Bridge):
    def __init__(self):
        super().__init__("fake", None)
        self.seen = None

    def answer(self, system, context, messages):
        self.seen = (system, context, messages)
        return "fake reply"


@pytest.fixture
def server():
    bridge = FakeBridge()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), cb.make_handler(bridge, {ORIGIN}))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", bridge
    srv.shutdown()


def _call(url, method="GET", body=None, origin=ORIGIN, ctype="application/json", extra=None):
    headers = {"Content-Type": ctype, **(extra or {})}
    if origin:
        headers["Origin"] = origin
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)  # noqa: S310 - local test server
    try:
        with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_chat_roundtrip_with_allowed_origin(server):
    url, bridge = server
    body = {"system": "S", "context": {"a": 1}, "messages": [{"role": "user", "content": "hi"}]}
    status, headers, data = _call(f"{url}/chat", "POST", body)
    assert status == 200 and json.loads(data) == {"reply": "fake reply"}
    assert headers["Access-Control-Allow-Origin"] == ORIGIN
    assert bridge.seen == ("S", {"a": 1}, [{"role": "user", "content": "hi"}])
    status, _, data = _call(f"{url}/health")
    assert status == 200 and json.loads(data)["backend"] == "fake"


def test_other_websites_and_non_json_are_refused(server):
    url, bridge = server
    body = {"system": "S", "context": {}, "messages": []}
    assert _call(f"{url}/chat", "POST", body, origin="https://evil.example")[0] == 403
    assert _call(f"{url}/chat", "POST", body, ctype="text/plain")[0] == 415
    assert _call(f"{url}/chat", "POST", {"nope": 1})[0] == 400
    assert bridge.seen is None  # nothing reached the CLI


def test_preflight_allows_private_network_for_allowed_origin(server):
    url, _ = server
    pre = {
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Private-Network": "true",
    }
    status, headers, _ = _call(f"{url}/chat", "OPTIONS", extra=pre)
    assert status == 204 and headers["Access-Control-Allow-Private-Network"] == "true"
    assert _call(f"{url}/chat", "OPTIONS", origin="https://evil.example", extra=pre)[0] == 403
