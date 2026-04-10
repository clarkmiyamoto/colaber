"""Constants and configuration for colaber."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# OAuth2 credentials – see README.md for how to obtain these.
COLAB_CLIENT_ID = os.environ.get("COLAB_CLIENT_ID", "")
COLAB_CLIENT_SECRET = os.environ.get("COLAB_CLIENT_SECRET", "")

OAUTH_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/colaboratory",
]

# Colab backend
COLAB_BACKEND = "https://colab.research.google.com"
CLIENT_AGENT_HEADER = "vscode"

# Token caching
TOKEN_CACHE_DIR = Path.home() / ".config" / "colaber"
TOKEN_CACHE_PATH = TOKEN_CACHE_DIR / "token.json"

# Runtime keep-alive interval in seconds
KEEP_ALIVE_INTERVAL = 60

# Accelerator types
ACCELERATOR_MAP = {
    "gpu": "GPU",
    "tpu": "TPU_V2",
    "cpu": "STANDARD",
}

GPU_TYPE_MAP = {
    "t4": "T4",
    "l4": "L4",
    "a100": "A100",
}

# Runtime assignment outcome codes
OUTCOME_SUCCESS = {0, 4}
OUTCOME_DENIED = 1
OUTCOME_QUOTA_EXCEEDED = 2
OUTCOME_DENYLISTED = 5

# Default directories to exclude from upload
UPLOAD_EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".ruff_cache"}

# Remote working directory on Colab
REMOTE_PROJECT_DIR = "/content/project"
