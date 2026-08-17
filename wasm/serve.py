#!/usr/bin/env python3
"""Serves the built site with caching turned off.

`python3 -m http.server` honours If-Modified-Since, which is fine for HTML but
not for this site: worker scripts are fetched through the normal HTTP cache, and
a plain reload frequently reuses the old az_worker.js or worker.js even after a
rebuild. The symptom is an engine that keeps reporting a bug you already fixed -
and because the wasm module and its JS glue are versioned together, a stale
worker paired with a fresh .wasm can also fail in ways that make no sense.

Every response here carries no-store, so a normal reload is always enough and
Ctrl+Shift+R is never required.

Usage, from anywhere:
    python3 wasm/serve.py [port]

Defaults to port 8000 and to ../build/wasm/web relative to this file.
"""

import functools
import http.server
import os
import socketserver
import sys

DEFAULT_PORT = 8000
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "wasm", "web")


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        # One line per request, without the date noise SimpleHTTPRequestHandler adds.
        sys.stderr.write("%s\n" % (fmt % args))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    root = os.path.normpath(ROOT)

    if not os.path.isdir(root):
        raise SystemExit(
            f"{root} does not exist - build the site first:\n"
            f"  source /path/to/emsdk/emsdk_env.sh && cd wasm && ./build.sh"
        )

    handler = functools.partial(NoCacheHandler, directory=root)
    # Without this, restarting the server on the same port fails for a minute
    # while the old socket sits in TIME_WAIT.
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving {root} on http://localhost:{port}/  (caching disabled)")
        print(f"  http://localhost:{port}/        NNUE engine and AlphaYugo")
        print(f"  http://localhost:{port}/az.html AlphaYugo alone, for debugging")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
