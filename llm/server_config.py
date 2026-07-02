"""
MLX LLM local server configuration and launch helper.
"""

import socket
import logging
import requests

logger = logging.getLogger(__name__)

# Server configuration defaults
HOST = "127.0.0.1"

# Try LM Studio default port (1234) first, then fallback to MLX server default (8080)
PORT = 1234
if not socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect_ex((HOST, 1234)) == 0:
    if socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect_ex((HOST, 8080)) == 0:
        PORT = 8080

BASE_URL = f"http://{HOST}:{PORT}/v1"
MODEL_NAME = "mlx-community/Qwen3.5-9B-MLX-4bit" # Default fallback, overwritten by active client

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if the local server port is already occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def check_server_health() -> bool:
    """Verify if either the LM Studio or MLX server is up and responding to API requests."""
    for p in [1234, 8080]:
        try:
            url = f"http://{HOST}:{p}/v1/models"
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                global PORT, BASE_URL
                PORT = p
                BASE_URL = f"http://{HOST}:{p}/v1"
                logger.info(f"Local LLM server detected on port {p}.")
                return True
        except Exception:
            pass
    return False

def get_launch_command() -> str:
    """Return the bash command to run the MLX server."""
    return f"python -m mlx_lm.server --model {MODEL_NAME} --host {HOST} --port {PORT}"

