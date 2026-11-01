import json
import threading
import unittest
import urllib.error
import urllib.request as rq

from recall.config import Settings
from recall.server import create_server


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s = Settings()
        s.rate_limit_per_min = 1000
        cls.srv, cls.pipe = create_server(s, port=0)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def call(self, method, path, body=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = rq.Request(self.base + path, data=data, method=method)
        try:
            with rq.urlopen(req) as r:
                return r.status, json.loads(r.read() or b"{}"), r.headers
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}"), e.headers

    def test_health_and_frontend(self):
        self.assertEqual(self.call("GET", "/api/health")[0], 200)
        with rq.urlopen(self.base + "/") as r:
            html = r.read().decode()
            self.assertIn("Recall", html)
            self.assertIn("default-src 'self'", r.headers["Content-Security-Policy"])

    def test_query_roundtrip(self):
        self.call("POST", "/api/cache/clear", {})
        a = self.call("POST", "/api/query", {"question": "How many API requests per minute are allowed?"})[1]
        b = self.call("POST", "/api/query", {"question": "What is the API request limit per minute?"})[1]
        self.assertEqual(a["source"], "llm")
        self.assertEqual(b["source"], "cache")
        self.assertIn("600", b["answer"])
        stats = self.call("GET", "/api/stats")[1]
        self.assertGreaterEqual(stats["llm_calls_avoided"], 1)

    def test_validation_errors(self):
        self.assertEqual(self.call("POST", "/api/query", {"question": 5})[0], 400)
        self.assertEqual(self.call("POST", "/api/query", raw=b"not json")[0], 400)
        self.assertEqual(self.call("POST", "/api/query", raw=b"[1,2]")[0], 400)
        self.assertEqual(self.call("POST", "/api/documents", {"title": "x"})[0], 400)
        self.assertEqual(self.call("POST", "/api/config", {"cache_threshold": 7})[0], 400)
        self.assertEqual(self.call("POST", "/api/config", {"entity_guard": "yes"})[0], 400)
        self.assertEqual(self.call("GET", "/api/nope")[0], 404)
        self.assertEqual(self.call("DELETE", "/api/documents/does-not-exist")[0], 404)

    def test_oversized_body_rejected(self):
        code, _, _ = self.call("POST", "/api/query", raw=b'{"question":"' + b"a" * 300_000 + b'"}')
        self.assertEqual(code, 413)

    def test_path_traversal_blocked(self):
        for p in ("/../run.py", "/%2e%2e/run.py", "/..%2frun.py"):
            self.assertEqual(self.call("GET", p)[0], 404, p)

    def test_document_lifecycle_and_config(self):
        code, doc, _ = self.call("POST", "/api/documents", {"title": "Temp Doc", "text": "The otter mascot is named Ollie."})
        self.assertEqual(code, 201)
        self.assertEqual(self.call("DELETE", f"/api/documents/{doc['id']}")[0], 200)
        code, cfg, _ = self.call("POST", "/api/config", {"cache_threshold": 0.7})
        self.assertEqual((code, cfg["cache_threshold"]), (200, 0.7))
        self.call("POST", "/api/config", {"cache_threshold": self.pipe.embedder.default_cache_threshold})

    def test_rate_limit(self):
        from recall.guardrails import RateLimiter
        from recall.server import make_handler
        from http.server import ThreadingHTTPServer
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.pipe, RateLimiter(2)))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{srv.server_address[1]}/api/query"
        codes = []
        for _ in range(4):
            try:
                codes.append(rq.urlopen(rq.Request(url, json.dumps({"question": "refund policy?"}).encode(), method="POST")).status)
            except urllib.error.HTTPError as e:
                codes.append(e.code)
        srv.shutdown()
        self.assertEqual(codes, [200, 200, 429, 429])


if __name__ == "__main__":
    unittest.main()
