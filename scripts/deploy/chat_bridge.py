#!/usr/bin/env python3
"""Local chat bridge: lets the web UI's chat use YOUR ChatGPT or Claude subscription.

Run it on the computer where your browser runs, with the Codex CLI or Claude Code installed and
signed in (`codex login` / `claude`). The UI's chat panel ("My subscription (local bridge)") sends
the question, the conversation and the prediction JSON to this server; it runs one non-interactive
CLI call and returns the answer. Your login never leaves your machine, and the model gets no
tools: Codex runs read-only and Claude without tools, both in an empty temporary folder.

Standard library only (Python 3.9+), so it runs without installing anything::

    python3 scripts/deploy/chat_bridge.py                      # auto: codex if installed, else claude
    python3 scripts/deploy/chat_bridge.py --backend claude --model sonnet
    python3 scripts/deploy/chat_bridge.py --allow-origin https://my-site.vercel.app

Endpoints (127.0.0.1:8765 by default): ``GET /health``, ``POST /chat`` with JSON
``{"system": str, "context": object, "messages": [{"role": "user"|"assistant", "content": str}]}``
-> ``{"reply": str}``. Only the allowed browser origins may call it (default: the UI's dev
server on localhost:5173), and requests must be JSON, so other websites cannot use your
subscription through it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
MAX_BODY = 512 * 1024  # bytes: the prediction JSON is ~5 KB; this leaves room for long chats
TIMEOUT_S = 180


def build_prompt(system: str, context: object, messages: list[dict], with_system: bool) -> str:
    """One self-contained prompt: (rules), the model output, the conversation, the instruction."""
    parts = []
    if with_system and system:
        parts.append(f"<rules>\n{system}\n</rules>")
    parts.append(f"<model_output>\n{json.dumps(context, indent=1)}\n</model_output>")
    lines = [
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
        for m in messages
    ]
    parts.append("<conversation>\n" + "\n\n".join(lines) + "\n</conversation>")
    parts.append(
        "Reply as the Assistant to the last User message, following the rules. "
        "Answer only; do not run commands or read files."
    )
    return "\n\n".join(parts)


def backend_command(backend: str, workdir: Path, system: str, model: str | None) -> list[str]:
    """The CLI call for one answer. The prompt goes to stdin."""
    if backend == "codex":
        cmd = [
            "codex", "exec", "--skip-git-repo-check", "--ephemeral", "--sandbox", "read-only",
            "--color", "never", "-C", str(workdir), "-o", str(workdir / "reply.txt"),
        ]  # fmt: skip
        return [*cmd, *(["-m", model] if model else []), "-"]
    if backend == "claude":
        cmd = [
            "claude", "-p", "--output-format", "text", "--tools", "", "--strict-mcp-config",
            "--no-session-persistence", "--system-prompt", system,
        ]  # fmt: skip
        return cmd + (["--model", model] if model else [])
    raise ValueError(f"unknown backend {backend!r}")


def pick_backend(choice: str) -> str:
    if choice != "auto":
        if not shutil.which(choice):
            raise SystemExit(f"`{choice}` is not on PATH; install and sign in to it first")
        return choice
    for name in ("codex", "claude"):
        if shutil.which(name):
            return name
    raise SystemExit("neither `codex` nor `claude` is on PATH; install one and sign in")


class Bridge:
    """Runs one CLI call at a time (a second request waits)."""

    def __init__(self, backend: str, model: str | None):
        self.backend = backend
        self.model = model
        self.lock = threading.Lock()

    def answer(self, system: str, context: object, messages: list[dict]) -> str:
        with self.lock, tempfile.TemporaryDirectory(prefix="chat-bridge-") as tmp:
            workdir = Path(tmp)
            prompt = build_prompt(system, context, messages, with_system=self.backend == "codex")
            proc = subprocess.run(  # noqa: S603 - fixed CLI, prompt on stdin
                backend_command(self.backend, workdir, system, self.model),
                input=prompt,
                capture_output=True,
                text=True,
                cwd=workdir,
                timeout=TIMEOUT_S,
            )
            reply_file = workdir / "reply.txt"
            reply = reply_file.read_text() if reply_file.exists() else proc.stdout
            if proc.returncode != 0 or not reply.strip():
                tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
                raise RuntimeError(
                    f"{self.backend} failed (exit {proc.returncode}): {' | '.join(tail)}"
                )
            return reply.strip()


def make_handler(bridge: Bridge, origins: set[str]):
    class Handler(BaseHTTPRequestHandler):
        def _cors(self) -> None:
            origin = self.headers.get("Origin", "")
            if origin in origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                # Chrome's Private Network Access: a page may call localhost only if allowed.
                self.send_header("Access-Control-Allow-Private-Network", "true")

        def _json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _origin_ok(self) -> bool:
            # Browsers always send Origin on cross-origin requests; refuse unknown ones so other
            # websites cannot spend your subscription. (curl without Origin is allowed.)
            origin = self.headers.get("Origin")
            return origin is None or origin in origins

        def do_OPTIONS(self) -> None:
            self.send_response(204 if self._origin_ok() else 403)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:
            if not self._origin_ok():
                return self._json(403, {"error": "origin not allowed"})
            if self.path != "/health":
                return self._json(404, {"error": "not found"})
            self._json(200, {"ok": True, "backend": bridge.backend, "model": bridge.model})

        def do_POST(self) -> None:
            if not self._origin_ok():
                return self._json(403, {"error": "origin not allowed"})
            if self.path != "/chat":
                return self._json(404, {"error": "not found"})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._json(415, {"error": "send JSON"})
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._json(413, {"error": "request too large"})
            try:
                req = json.loads(self.rfile.read(length))
                messages = [m for m in req["messages"] if isinstance(m, dict)]
                reply = bridge.answer(str(req.get("system", "")), req.get("context", {}), messages)
            except (ValueError, KeyError, TypeError):
                return self._json(400, {"error": "expected {system, context, messages}"})
            except subprocess.TimeoutExpired:
                return self._json(
                    504, {"error": f"{bridge.backend} took longer than {TIMEOUT_S} s"}
                )
            except RuntimeError as exc:
                return self._json(502, {"error": str(exc)})
            self._json(200, {"reply": reply})

        def log_message(self, fmt: str, *args) -> None:  # one short line per request
            print(f"[bridge] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", choices=["auto", "codex", "claude"], default="auto")
    parser.add_argument("--model", help="model name passed to the CLI (default: the CLI's own)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-origin", action="append", default=[], help="extra browser origin (repeatable)"
    )
    args = parser.parse_args()

    bridge = Bridge(pick_backend(args.backend), args.model)
    origins = {*DEFAULT_ORIGINS, *args.allow_origin}
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(bridge, origins))
    print(f"chat bridge: {bridge.backend}{f' ({bridge.model})' if bridge.model else ''} on "
          f"http://127.0.0.1:{args.port} for {', '.join(sorted(origins))}. Ctrl+C to stop.")  # fmt: skip
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
