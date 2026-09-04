#!/usr/bin/env python3
"""Static server for the front-end.

Sends the cross-origin isolation headers that let onnxruntime-web use
multithreaded WASM (noticeably faster). `credentialless` keeps the CDN load
of onnxruntime working.

    python3 web/serve.py        then open http://localhost:8000
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".onnx": "application/octet-stream",
        ".wasm": "application/wasm",
        ".mjs": "text/javascript",
    }

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "200" not in fmt % args:
            super().log_message(fmt, *args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dir", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()

    handler = partial(Handler, directory=str(args.dir.resolve()))
    print(f"Serving http://localhost:{args.port}  (root: {args.dir.resolve()})")
    ThreadingHTTPServer(("0.0.0.0", args.port), handler).serve_forever()
