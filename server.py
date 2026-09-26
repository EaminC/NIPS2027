"""Local workbench for the traffic-layer AB/AA lab. Python plus numpy."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ab_aa_lab.scenarios import catalog, run_form  # noqa: E402

STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        del fmt, args

    def send_bytes(self, payload: bytes, status: int = 200, content_type: str = "application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, value, status: int = 200):
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode(), status)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/catalog":
            return self.send_json(catalog())
        static = STATIC.get(url.path)
        if not static:
            return self.send_json({"error": "Not found"}, 404)
        name, content_type = static
        self.send_bytes((ROOT / "static" / name).read_bytes(), content_type=content_type)

    def do_POST(self):
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin and origin != "http://" + host:
            return self.send_json({"error": "Origin rejected"}, 403)
        if urlparse(self.path).path != "/api/run":
            return self.send_json({"error": "Not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 100_000:
                raise ValueError("请求过大")
            form = json.loads(self.rfile.read(length) or b"{}")
            self.send_json(run_form(form))
        except (ValueError, KeyError, TypeError) as exc:
            self.send_json({"error": str(exc)}, 400)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"流量层实验室: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
