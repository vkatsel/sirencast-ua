"""
Zero-dependency lightweight HTTP server for SirenCast UA.
Serves live forecasting API and modern web dashboard using standard library http.server.
"""
import http.server
import json
import logging
import socketserver
import sys
from pathlib import Path

from src.config import BASE_DIR
from src.live_forecast import predict_live_risk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WEB_DIR = BASE_DIR / "web"
INDEX_HTML_PATH = WEB_DIR / "index.html"
PORT = 8000


class SirenCastHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """
    Handles API endpoints and serves static web UI.
    """
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self.serve_file(INDEX_HTML_PATH, "text/html; charset=utf-8")
        elif self.path.startswith("/api/forecast"):
            self.serve_forecast_api()
        elif self.path == "/api/health":
            self.send_json_response({"status": "ok", "service": "SirenCast-UA"})
        else:
            # Default to 404 or index
            self.serve_file(INDEX_HTML_PATH, "text/html; charset=utf-8")

    def serve_forecast_api(self):
        try:
            forecast = predict_live_risk()
            # Convert timestamp to isoformat string for clean JSON serialization
            if "metadata" in forecast and "timestamp_utc" in forecast["metadata"]:
                forecast["metadata"]["timestamp_utc"] = forecast["metadata"]["timestamp_utc"].isoformat()
            self.send_json_response(forecast)
        except Exception as e:
            logger.error(f"Error computing live forecast: {e}")
            self.send_json_response({"error": str(e)}, status_code=500)

    def serve_file(self, file_path: Path, content_type: str):
        if not file_path.exists():
            self.send_error(404, "File Not Found")
            return

        try:
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            logger.error(f"Error serving {file_path}: {e}")
            self.send_error(500, "Internal Server Error")

    def send_json_response(self, data: dict, status_code: int = 200):
        try:
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            logger.error(f"Error sending json: {e}")

    def log_message(self, format, *args):
        # Concise logging
        logger.info(f"{self.address_string()} - {format % args}")


def start_server(port: int = PORT):
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    server_address = ("", port)
    
    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(server_address, SirenCastHTTPRequestHandler) as httpd:
        print("\n" + "=" * 65)
        print(f"SIRENCAST UA - LIVE WEB DASHBOARD SERVER")
        print(f"Server running at: http://localhost:{port}/")
        print(f"API endpoint:      http://localhost:{port}/api/forecast")
        print("Press Ctrl+C to stop.")
        print("=" * 65 + "\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server gracefully...")
            httpd.shutdown()


if __name__ == "__main__":
    start_server()
