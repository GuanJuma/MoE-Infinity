# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Minimal OpenAI-compatible server for the Hy3 engine (stdlib only).

Lets HTTP clients such as tokenhub_bench drive Hy3 on MoE-Infinity and
writes one routing trace per request::

    python -m moe_infinity.hy3.server --model-dir /model \
        --offload-dir /scratch/$ME/hy3_store --port 30000 \
        --trace-dir /scratch/$ME/hy3_traces/server

Endpoints: ``GET /v1/models``, ``GET /health``, ``POST /v1/chat/completions``
(``stream`` with ``stream_options.include_usage``, ``chat_template_kwargs``,
``max_tokens``/``max_completion_tokens``, ``temperature``, ``top_p``,
``seed``), ``POST /flush_cache`` (no-op: there is no prefix cache, so
``usage.prompt_tokens_details.cached_tokens`` is always 0).  Requests are
served one at a time (batch size 1); concurrent clients queue.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .cli import add_engine_args, config_from_args
from .engine import Hy3Engine
from .routing_trace import save_trace


class _State:
    engine: Hy3Engine = None
    lock = threading.Lock()
    served_name = "/model"
    api_key = None
    trace_dir = None
    seq = 0
    default_max_tokens = 512


def _usage(stats: dict) -> dict:
    p, c = stats["prompt_tokens"], stats["completion_tokens"]
    return {
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": p + c,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "moe-infinity-hy3"

    def log_message(self, fmt, *args):  # quieter default logging
        print(f"[hy3-server] {self.address_string()} {fmt % args}", flush=True)

    # -- helpers ----------------------------------------------------------

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, msg: str, typ: str = "invalid_request_error"):
        self._json(code, {"error": {"message": msg, "type": typ, "code": code}})

    def _authorized(self) -> bool:
        if not _State.api_key:
            return True
        return self.headers.get("Authorization", "") == (
            f"Bearer {_State.api_key}"
        )

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # -- routes -----------------------------------------------------------

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            return self._json(200, {"status": "ok"})
        if not self._authorized():
            return self._error(401, "invalid api key", "authentication_error")
        if self.path.rstrip("/") == "/v1/models":
            eng = _State.engine
            return self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": _State.served_name,
                            "object": "model",
                            "owned_by": "moe-infinity",
                            "max_model_len": eng.cfg.max_context,
                        }
                    ],
                },
            )
        return self._error(404, f"no route {self.path}")

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/flush_cache":
            return self._json(200, {"ok": True, "note": "no prefix cache"})
        if not self._authorized():
            return self._error(401, "invalid api key", "authentication_error")
        if path != "/v1/chat/completions":
            return self._error(404, f"no route {self.path}")
        try:
            req = self._body()
        except Exception as e:  # noqa: BLE001
            return self._error(400, f"bad json: {e}")
        msgs = req.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return self._error(400, "messages must be a non-empty list")
        try:
            self._chat(req, msgs)
        except ValueError as e:
            self._error(400, str(e))
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._error(500, f"{type(e).__name__}: {e}", "server_error")

    def _chat(self, req: dict, msgs: list) -> None:
        eng = _State.engine
        kw = dict(req.get("chat_template_kwargs") or {})
        if "reasoning_effort" in req and "reasoning_effort" not in kw:
            kw["reasoning_effort"] = req["reasoning_effort"]
        max_tokens = int(
            req.get("max_completion_tokens")
            or req.get("max_tokens")
            or _State.default_max_tokens
        )
        ids = eng.encode_chat(msgs, kw)
        max_tokens = max(1, min(max_tokens, eng.cfg.max_context - len(ids)))
        if max_tokens <= 0 or len(ids) >= eng.cfg.max_context:
            raise ValueError(
                f"prompt is {len(ids)} tokens; max_context is "
                f"{eng.cfg.max_context}"
            )
        stream = bool(req.get("stream"))
        include_usage = bool(
            (req.get("stream_options") or {}).get("include_usage")
        )
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        gen_kw = dict(
            max_new_tokens=max_tokens,
            temperature=float(req.get("temperature") or 0.0),
            top_p=float(req.get("top_p") or 1.0),
            seed=int(req.get("seed") or 0),
            ignore_eos=bool(req.get("ignore_eos", False)),
        )
        tok = eng.tokenizer

        with _State.lock:
            _State.seq += 1
            seq = _State.seq
            rid = f"{seq:06d}_{cid[-8:]}"
            meta = {
                "source": "server",
                "request": {
                    k: req.get(k)
                    for k in (
                        "max_tokens",
                        "temperature",
                        "chat_template_kwargs",
                    )
                },
            }
            trace = _State.trace_dir is not None
            if not stream:
                r = eng.generate(ids, rid=rid, meta=meta, trace=trace, **gen_kw)
                text = tok.decode(r["output_ids"], skip_special_tokens=True)
                self._save(r)
                return self._json(
                    200,
                    {
                        "id": cid,
                        "object": "chat.completion",
                        "created": created,
                        "model": _State.served_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": text,
                                },
                                "finish_reason": r["stats"]["finish_reason"],
                            }
                        ],
                        "usage": _usage(r["stats"]),
                    },
                )

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def chunk(delta, finish=None, usage=None, choices=True):
                obj = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": _State.served_name,
                    "choices": [
                        {"index": 0, "delta": delta, "finish_reason": finish}
                    ]
                    if choices
                    else [],
                }
                if usage is not None:
                    obj["usage"] = usage
                self.wfile.write(
                    b"data: "
                    + json.dumps(obj, ensure_ascii=False).encode()
                    + b"\n\n"
                )
                self.wfile.flush()

            chunk({"role": "assistant", "content": ""})
            out, sent = [], [""]

            def on_token(t):
                out.append(t)
                text = tok.decode(out, skip_special_tokens=True)
                if text.endswith("\ufffd"):
                    return
                new = text[len(sent[0]) :]
                if new:
                    sent[0] = text
                    chunk({"content": new})

            r = eng.generate(
                ids,
                rid=rid,
                meta=meta,
                trace=trace,
                on_token=on_token,
                **gen_kw,
            )
            text = tok.decode(out, skip_special_tokens=True)
            if len(text) > len(sent[0]):
                chunk({"content": text[len(sent[0]) :]})
            chunk({}, finish=r["stats"]["finish_reason"])
            if include_usage:
                chunk(None, usage=_usage(r["stats"]), choices=False)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self._save(r)

    def _save(self, r: dict) -> None:
        if _State.trace_dir is None or "trace" not in r:
            return
        save_trace(r["trace"], _State.trace_dir / f"{r['rid']}.pt")
        with open(_State.trace_dir / "requests.jsonl", "a") as f:
            f.write(json.dumps({"rid": r["rid"], **r["stats"]}) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_engine_args(p)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--served-model-name", default="/model")
    p.add_argument("--api-key", default=None)
    p.add_argument("--trace-dir", default=None)
    p.add_argument("--default-max-tokens", type=int, default=512)
    a = p.parse_args(argv)
    _State.engine = Hy3Engine(config_from_args(a))
    print(json.dumps(_State.engine.memory_report(), indent=1), flush=True)
    _State.served_name = a.served_model_name
    _State.api_key = a.api_key
    _State.default_max_tokens = a.default_max_tokens
    if a.trace_dir:
        _State.trace_dir = Path(a.trace_dir)
        _State.trace_dir.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(
        f"[hy3-server] ready on http://{a.host}:{a.port}/v1 "
        f"(model id {a.served_model_name!r})",
        flush=True,
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        from .experts import exit_process

        exit_process(0)


if __name__ == "__main__":
    main()
