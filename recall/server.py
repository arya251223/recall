"""Zero-dependency HTTP server (stdlib only): JSON API + static frontend."""
from __future__ import annotations

import json
import mimetypes
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .config import ROOT, Settings
from .guardrails import RateLimiter
from .pipeline import Pipeline, build_pipeline

WEB = ROOT / "web"
MAX_BODY = 200_000
MAX_DOCS = 200
MAX_DOC_CHARS = 50_000


def make_handler(pipeline: Pipeline, limiter: RateLimiter):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Recall/1.0"

        def log_message(self, fmt, *args):  # quiet by default
            if "--verbose" in sys.argv:
                super().log_message(fmt, *args)

        # -- helpers --------------------------------------------------------------
        def _send(self, code: int, payload, ctype: str = "application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if ctype.startswith("text/html"):
                self.send_header("Content-Security-Policy",
                                 "default-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
            self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _json_body(self) -> dict | None:
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = -1
            if n < 0 or n > MAX_BODY:
                self._send(413, {"error": "Request body too large."})
                return None
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(data, dict):
                    raise ValueError
                return data
            except ValueError:
                self._send(400, {"error": "Body must be a JSON object."})
                return None

        # -- routes ---------------------------------------------------------------
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/health":
                return self._send(200, {"status": "ok", **pipeline.config_view()})
            if path == "/api/stats":
                return self._send(200, {**pipeline.metrics.snapshot(), "cache": pipeline.cache.counters,
                                        "config": pipeline.config_view()})
            if path == "/api/documents":
                return self._send(200, {"documents": pipeline.kb.listing()})
            if path == "/api/cache":
                return self._send(200, {"entries": pipeline.cache.view()})
            if path.startswith("/api/"):
                return self._send(404, {"error": "Unknown endpoint."})
            return self._static(path)

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/api/query":
                if not limiter.allow(self.client_address[0]):
                    return self._send(429, {"error": "Too many requests. Slow down."})
                body = self._json_body()
                if body is None:
                    return
                q = body.get("question")
                if not isinstance(q, str):
                    return self._send(400, {"error": "'question' must be a string."})
                return self._send(200, pipeline.query(q, use_cache=bool(body.get("use_cache", True))))
            if path == "/api/documents":
                body = self._json_body()
                if body is None:
                    return
                title, text = body.get("title"), body.get("text")
                if not isinstance(title, str) or not isinstance(text, str) or not text.strip():
                    return self._send(400, {"error": "'title' and non-empty 'text' are required."})
                if len(text) > MAX_DOC_CHARS:
                    return self._send(413, {"error": f"Document exceeds {MAX_DOC_CHARS} characters."})
                if len(pipeline.kb.docs) >= MAX_DOCS:
                    return self._send(409, {"error": "Document limit reached."})
                return self._send(201, pipeline.add_document(title, text))
            if path == "/api/cache/clear":
                return self._send(200, {"cleared": pipeline.cache.clear()})
            if path == "/api/config":
                body = self._json_body()
                if body is None:
                    return
                err = self._apply_config(body)
                return self._send(400, {"error": err}) if err else self._send(200, pipeline.config_view())
            return self._send(404, {"error": "Unknown endpoint."})

        def do_DELETE(self):
            m = re.fullmatch(r"/api/documents/([a-z0-9\-]+)", urlparse(self.path).path)
            if not m:
                return self._send(404, {"error": "Unknown endpoint."})
            res = pipeline.remove_document(m.group(1))
            return self._send(200 if res["removed"] else 404, res)

        def _apply_config(self, body: dict) -> str | None:
            if "cache_threshold" in body:
                v = body["cache_threshold"]
                if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0.0 <= v <= 1.0:
                    return "cache_threshold must be a number between 0 and 1."
                pipeline.cache.threshold = float(v)
            for key, setter in (("entity_guard", lambda b: setattr(pipeline.cache, "entity_guard", b)),
                                ("consistency_check", lambda b: setattr(pipeline.cfg, "consistency_check", b))):
                if key in body:
                    if not isinstance(body[key], bool):
                        return f"{key} must be a boolean."
                    setter(body[key])
            return None

        def _static(self, path: str):
            rel = "index.html" if path in ("/", "") else unquote(path).lstrip("/")
            target = (WEB / rel).resolve()
            if WEB.resolve() not in target.parents and target != WEB.resolve() or not target.is_file():
                return self._send(404, {"error": "Not found."})
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript",):
                ctype += "; charset=utf-8"
            self._send(200, target.read_bytes(), ctype)

    return Handler


def create_server(settings: Settings | None = None, pipeline: Pipeline | None = None, port: int | None = None):
    settings = settings or Settings()
    pipeline = pipeline or build_pipeline(settings)
    handler = make_handler(pipeline, RateLimiter(settings.rate_limit_per_min))
    return ThreadingHTTPServer((settings.host, settings.port if port is None else port), handler), pipeline


def main():
    settings = Settings()
    srv, pipe = create_server(settings)
    v = pipe.config_view()
    print(f"Recall running on http://{settings.host}:{srv.server_address[1]}")
    print(f"  embedder={v['embedder']}  llm={v['llm']}  docs={v['kb_docs']}  threshold={v['cache_threshold']}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
