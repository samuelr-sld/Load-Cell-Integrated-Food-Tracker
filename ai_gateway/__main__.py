"""`python -m ai_gateway [--host IP] [--port 8100]`: serve the PlateGo AI gateway on the laptop's Wi-Fi address."""
import argparse
import logging
import socket

import httpx
import uvicorn

from .app import Settings, create_app


def lan_ip() -> str:
    """The address this laptop uses on its default route (no packet is sent). Works without internet if a router is present."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("192.168.0.1", 9))
            return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=None, help="interface to bind (default: the Wi-Fi/LAN address)")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    s = Settings()
    host = args.host or lan_ip()
    if host in ("0.0.0.0", "::"):
        raise SystemExit("Bind to one interface (the Wi-Fi address), not all of them.")

    # Load the model now so the first question is not slow, and report problems before the phone sees them.
    try:
        r = httpx.post(f"{s.ollama_url}/api/generate", json={"model": s.model, "keep_alive": s.keep_alive}, timeout=120)
        status = "loaded" if r.status_code == 200 else f"NOT available ({r.status_code}): run  ollama pull {s.model}"
    except httpx.HTTPError:
        status = "NOT available: Ollama is not running (start the Ollama app)"

    print(f"\n  PlateGo AI gateway\n  Address for the app : http://{host}:{args.port}\n  Pairing token       : {s.token}"
          f"\n  Model               : {s.model} ({status})\n", flush=True)
    uvicorn.run(create_app(s), host=host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
