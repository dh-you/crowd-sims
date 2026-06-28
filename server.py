"""Serve the browser simulations from the repository root."""

from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

PORT = 1337

Handler = SimpleHTTPRequestHandler
Handler.extensions_map.update({
        '': 'application/octet-stream',
        '.manifest': 'text/cache-manifest',
        '.html': 'text/html',
        '.png': 'image/png',
        '.jpg': 'image/jpg',
        '.svg': 'image/svg+xml',
        '.css': 'text/css',
        '.js': 'application/javascript',
        '.wasm': 'application/wasm',
        '.json': 'application/json',
        '.xml': 'application/xml',
})

def run(server_class=ThreadingHTTPServer, handler_class=Handler):
    server_address = ('', PORT)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()


if __name__ == "__main__":
    try:
        print("Server starting on http://localhost:1337/")
        run()
    except KeyboardInterrupt:
        print("\nServer stopped.")
