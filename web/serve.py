#!/usr/bin/env python3
"""Static server for the front-end.

Two things the stock handler does not do:

- cross-origin isolation headers, which let onnxruntime-web use multithreaded
  WASM (noticeably faster); `credentialless` keeps the CDN load working;
- HTTP range requests, without which a browser cannot seek inside a video.

    python3 web/serve.py        then open http://localhost:8000
"""
import argparse
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
CHUNK = 64 * 1024


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
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_head(self):
        self._remaining = None
        header = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not header or os.path.isdir(path):
            return super().send_head()

        match = RANGE_RE.fullmatch(header.strip())
        if not match:
            self.send_error(400, "Malformed Range header")
            return None
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(f.fileno()).st_size
        first, last = match.groups()
        if first == "":                      # suffix range: last N bytes
            start, end = max(size - int(last or 0), 0), size - 1
        else:
            start = int(first)
            end = int(last) if last else size - 1

        if start >= size:
            f.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        end = min(end, size - 1)
        f.seek(start)
        self._remaining = end - start + 1

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(self._remaining))
        self.end_headers()
        return f

    def copyfile(self, source, outputfile):
        if self._remaining is None:
            return super().copyfile(source, outputfile)
        remaining = self._remaining
        while remaining > 0:
            chunk = source.read(min(CHUNK, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def log_message(self, fmt, *args):
        if not any(code in fmt % args for code in ("200", "206")):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dir", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()

    handler = partial(Handler, directory=str(args.dir.resolve()))
    print(f"Serving http://localhost:{args.port}  (root: {args.dir.resolve()})")
    ThreadingHTTPServer(("0.0.0.0", args.port), handler).serve_forever()
