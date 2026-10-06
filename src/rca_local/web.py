"""Loopback-only local browser UI; access remotely through an SSH tunnel."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import urlparse

from .core import observation_from_form
from .pipeline import run_pipeline
from .workers import settings

TOKEN = secrets.token_urlsafe(32)
BUSY = threading.Lock()
JOBS = {}
JOBS_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def allowed_host(self):
        return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

    def send(self, code, body, content_type="application/json"):
        data = body.encode() if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.allowed_host():
            return self.send(403, {"error": "Use the localhost address printed by the server."})
        path = urlparse(self.path).path
        if path == "/":
            page = Path(__file__).with_name("index.html").read_text()
            return self.send(200, page.replace("__RCA_TOKEN__", TOKEN), "text/html")
        if path == "/api/example":
            root, _, _, _ = settings()
            try:
                current = json.loads((root / "syn_data/product_a_scenarios_v1/case_07/observation.json").read_text())
                return self.send(200, current)
            except OSError:
                return self.send(404, {"error": "Case 07 is unavailable in RCA_DATA_ROOT."})
        if path.startswith("/api/jobs/"):
            with JOBS_LOCK:
                job = JOBS.get(path.rsplit("/", 1)[-1])
            return self.send(200 if job else 404, job or {"error": "Unknown request."})
        return self.send(404, {"error": "Not found."})

    def do_POST(self):
        if not self.allowed_host() or self.headers.get("X-RCA-Token") != TOKEN:
            return self.send(403, {"error": "Refresh the local page before submitting."})
        if self.path != "/api/analyze":
            return self.send(404, {"error": "Not found."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 64000:
                raise ValueError("Request is empty or too large.")
            form = json.loads(self.rfile.read(length))
            if not isinstance(form, dict):
                raise ValueError("Expected an input form.")
            current = observation_from_form(form)
        except (ValueError, TypeError) as exc:
            return self.send(400, {"error": str(exc)})
        if not BUSY.acquire(blocking=False):
            return self.send(409, {"error": "One investigation is already running. Wait for it to finish."})
        job_id = secrets.token_hex(12)
        with JOBS_LOCK:
            if len(JOBS) >= 32:
                JOBS.pop(next(iter(JOBS)))
            JOBS[job_id] = {"state": "running"}

        def execute():
            try:
                answer = run_pipeline(current)
                result = {"state": "done", "answer": answer}
            except Exception as exc:
                result = {"state": "error", "error": str(exc)}
            finally:
                BUSY.release()
            with JOBS_LOCK:
                JOBS[job_id] = result

        threading.Thread(target=execute, daemon=True).start()
        return self.send(202, {"job_id": job_id})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Airgap RCA: http://127.0.0.1:{server.server_port}", flush=True)
    print("Use SSH port forwarding from your laptop. Ctrl+C stops the app.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
