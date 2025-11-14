import os
import signal
import subprocess
import sys
import time

import psutil
import pytest
import requests

PORT = 9001
SERVER_START_TIMEOUT_S = 40


def kill_process_on_port(port: int) -> None:
    """Terminate any process listening on the given port."""
    for proc in psutil.process_iter(attrs=["pid", "name"]):
        try:
            for conn in proc.connections(kind="inet"):
                if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
                    proc.kill()
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


@pytest.fixture(scope="session")
def gradio_server():
    kill_process_on_port(PORT)
    process = subprocess.Popen(
        [
            "uv",
            "run",
            "src/gradio/qual/trajectory_visualizer.py",
            "--no-share",
            "--port",
            str(PORT),
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        universal_newlines=True,
        preexec_fn=os.setsid,
    )
    print(
        "Waiting for Gradio to start... "
        f"Waiting a maximum of {SERVER_START_TIMEOUT_S} seconds... "
        "Please do not interrupt the process early!"
    )
    server_url = f"http://localhost:{PORT}"
    start = time.time()
    while True:
        if process.poll() is not None:
            raise RuntimeError("Gradio server failed to start")
        try:
            response = requests.get(server_url, timeout=1)
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            pass
        if time.time() - start > SERVER_START_TIMEOUT_S:
            raise RuntimeError("Gradio server failed to start in time")
        time.sleep(1)
    try:
        yield server_url
    finally:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait()
