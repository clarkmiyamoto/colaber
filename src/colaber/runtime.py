"""Colab runtime assignment, keep-alive, and teardown."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass

import requests
from google.oauth2.credentials import Credentials

from colaber.config import (
    ACCELERATOR_MAP,
    CLIENT_AGENT_HEADER,
    COLAB_BACKEND,
    GPU_TYPE_MAP,
    KEEP_ALIVE_INTERVAL,
    OUTCOME_DENIED,
    OUTCOME_DENYLISTED,
    OUTCOME_QUOTA_EXCEEDED,
    OUTCOME_SUCCESS,
)


def _parse_xssi_json(text: str) -> dict:
    """Parse a Colab API response, stripping the anti-XSSI prefix `)]}'\n`."""
    text = text.strip()
    if text.startswith(")]}'"):
        text = text[4:].lstrip("\n")
    return json.loads(text)


class ColabRuntimeError(Exception):
    """Error during Colab runtime operations."""


class QuotaExceededError(ColabRuntimeError):
    """Colab compute quota has been exceeded."""


class RuntimeDeniedError(ColabRuntimeError):
    """Runtime assignment was denied."""


@dataclass
class RuntimeInfo:
    """Information about an assigned Colab runtime."""

    endpoint: str
    proxy_token: str
    proxy_url: str


def _generate_notebook_hash() -> str:
    """Generate a fake notebook hash for runtime assignment.

    Format: UUID with dashes replaced by underscores, padded to 44 chars with dots.
    """
    return str(uuid.uuid4()).replace("-", "_").ljust(44, ".")


class ColabRuntime:
    """Manages a Colab runtime lifecycle.

    Use as a context manager to ensure cleanup:

        with ColabRuntime(credentials, accelerator="gpu") as rt:
            # rt.info has endpoint, proxy_token, proxy_url
            ...
    """

    def __init__(
        self,
        credentials: Credentials,
        accelerator: str = "gpu",
        gpu_type: str = "t4",
    ):
        self.credentials = credentials
        self.accelerator = ACCELERATOR_MAP.get(accelerator, accelerator)
        self.gpu_type = GPU_TYPE_MAP.get(gpu_type, gpu_type)
        self.info: RuntimeInfo | None = None
        self._keepalive_stop = threading.Event()
        self._keepalive_thread: threading.Thread | None = None
        self._session = requests.Session()

    def _headers(self, xsrf_token: str | None = None) -> dict[str, str]:
        """Build request headers."""
        headers = {
            "Authorization": f"Bearer {self.credentials.token}",
            "X-Colab-Client-Agent": CLIENT_AGENT_HEADER,
            "Accept": "application/json",
        }
        if xsrf_token:
            headers["X-Goog-Colab-Token"] = xsrf_token
            headers["Content-Type"] = "application/json"
        return headers

    def _assign_params(self) -> dict[str, str]:
        """Build query parameters for the assign request."""
        params = {
            "nbh": _generate_notebook_hash(),
            "variant": self.accelerator,
            "authuser": "0",
        }
        if self.accelerator == "GPU":
            params["accelerator"] = self.gpu_type
        return params

    def _unassign_endpoint(self, endpoint: str) -> None:
        """Try to unassign a specific runtime endpoint."""
        url = f"{COLAB_BACKEND}/tun/m/unassign/{endpoint}"
        try:
            r = self._session.get(url, headers=self._headers())
            if r.status_code == 200:
                xsrf = _parse_xssi_json(r.text).get("token", "")
                self._session.post(url, headers=self._headers(xsrf))
        except Exception:
            pass

    def _release_existing_runtime(self, error_resp: requests.Response) -> None:
        """Try to release an existing runtime that's causing a 412 conflict.

        Attempts multiple strategies to find and release the active runtime.
        """
        # Strategy 1: Parse the 412 response body for endpoint info
        try:
            data = _parse_xssi_json(error_resp.text)
            endpoint = data.get("endpoint") or data.get("id")
            if endpoint:
                self._unassign_endpoint(endpoint)
                return
        except Exception:
            pass

        # Strategy 2: Try the Colab sessions/runtimes listing endpoints
        for path in ("/api/colabx/sessions", "/api/sessions"):
            try:
                resp = self._session.get(
                    f"{COLAB_BACKEND}{path}", headers=self._headers()
                )
                if resp.status_code != 200:
                    continue
                sessions = _parse_xssi_json(resp.text)
                if isinstance(sessions, dict):
                    sessions = list(sessions.values()) if sessions else []
                if not isinstance(sessions, list):
                    continue
                for session in sessions:
                    if isinstance(session, dict):
                        ep = session.get("endpoint") or session.get("id")
                        if ep:
                            self._unassign_endpoint(ep)
            except Exception:
                continue

    def assign(self) -> RuntimeInfo:
        """Request a Colab runtime with the specified accelerator.

        Performs the GET-then-POST XSRF token dance required by the backend.
        If a 412 is returned (existing runtime), tries to release it and retry.
        """
        url = f"{COLAB_BACKEND}/tun/m/assign"

        for attempt in range(3):
            params = self._assign_params()

            # Step 1: GET to obtain XSRF token
            resp = self._session.get(url, params=params, headers=self._headers())
            resp.raise_for_status()
            xsrf_token = _parse_xssi_json(resp.text)["token"]

            # Step 2: POST with XSRF token to actually assign
            resp = self._session.post(
                url, params=params, headers=self._headers(xsrf_token)
            )

            if resp.status_code == 412:
                if attempt < 2:
                    self._release_existing_runtime(resp)
                    time.sleep(2)  # Brief pause before retry
                    continue
                raise ColabRuntimeError(
                    "A Colab runtime is already active. "
                    "Go to colab.research.google.com, click 'Manage Sessions' "
                    "in the Runtime menu, and terminate existing runtimes."
                )

            break

        resp.raise_for_status()
        data = _parse_xssi_json(resp.text)

        # Check for error outcomes (field only present on failure)
        outcome = data.get("outcome")
        if outcome is not None and outcome not in OUTCOME_SUCCESS:
            if outcome == OUTCOME_DENIED:
                raise RuntimeDeniedError(
                    "Runtime denied. The requested accelerator may be unavailable."
                )
            if outcome == OUTCOME_QUOTA_EXCEEDED:
                raise QuotaExceededError(
                    "Compute quota exceeded. Try again later or upgrade your Colab plan."
                )
            if outcome == OUTCOME_DENYLISTED:
                raise RuntimeDeniedError("Your account has been restricted from Colab runtimes.")
            raise ColabRuntimeError(f"Unexpected assignment outcome: {outcome}")

        if "endpoint" not in data:
            raise ColabRuntimeError(f"No endpoint in assignment response: {data}")

        proxy_info = data.get("runtimeProxyInfo", {})
        self.info = RuntimeInfo(
            endpoint=data["endpoint"],
            proxy_token=proxy_info["token"],
            proxy_url=proxy_info["url"],
        )
        return self.info

    def unassign(self) -> None:
        """Release the assigned runtime."""
        if not self.info:
            return

        url = f"{COLAB_BACKEND}/tun/m/unassign/{self.info.endpoint}"

        try:
            # GET for XSRF token
            resp = self._session.get(url, headers=self._headers())
            resp.raise_for_status()
            xsrf_token = _parse_xssi_json(resp.text)["token"]

            # POST to unassign
            self._session.post(url, headers=self._headers(xsrf_token))
        except Exception:
            pass  # Best-effort cleanup

        self.info = None

    def keep_alive(self) -> None:
        """Send a single keep-alive ping."""
        if not self.info:
            return
        url = f"{COLAB_BACKEND}/tun/m/{self.info.endpoint}/keep-alive/"
        try:
            self._session.get(url, headers=self._headers())
        except Exception:
            pass  # Non-fatal

    def _keepalive_loop(self) -> None:
        """Background loop that pings keep-alive every KEEP_ALIVE_INTERVAL seconds."""
        while not self._keepalive_stop.wait(KEEP_ALIVE_INTERVAL):
            self.keep_alive()

    def _start_keepalive(self) -> None:
        """Start the keep-alive background thread."""
        self._keepalive_stop.clear()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop, daemon=True
        )
        self._keepalive_thread.start()

    def _stop_keepalive(self) -> None:
        """Stop the keep-alive background thread."""
        self._keepalive_stop.set()
        if self._keepalive_thread:
            self._keepalive_thread.join(timeout=5)
            self._keepalive_thread = None

    def __enter__(self) -> ColabRuntime:
        self.assign()
        self._start_keepalive()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop_keepalive()
        self.unassign()
        self._session.close()
