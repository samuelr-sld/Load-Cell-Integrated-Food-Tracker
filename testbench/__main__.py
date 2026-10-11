"""`python -m testbench` -> serve the testbench on http://127.0.0.1:8000 (host/port: TESTBENCH_HOST / TESTBENCH_PORT)."""
import logging

import uvicorn

from .config import Settings

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = Settings()
    uvicorn.run("testbench.main:app", host=s.host, port=s.port, log_level="info")
