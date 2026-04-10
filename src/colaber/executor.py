"""Execute Python code on a Colab runtime via Jupyter kernel WebSocket protocol."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from dataclasses import dataclass, field
from typing import Callable

import requests
import websockets

import getpass

from colaber.config import CLIENT_AGENT_HEADER
from colaber.runtime import ColabRuntime


def _default_stdin(prompt: str, password: bool) -> str:
    """Default stdin handler: read from terminal, masking input for passwords."""
    if password:
        return getpass.getpass(prompt)
    return input(prompt)


@dataclass
class ExecutionResult:
    """Result of a code execution on the remote kernel."""

    status: str  # "ok" or "error"
    error_name: str | None = None
    error_value: str | None = None
    traceback: list[str] = field(default_factory=list)


def _jupyter_message(
    msg_type: str,
    content: dict,
    session_id: str,
) -> str:
    """Build a Jupyter wire protocol message as JSON."""
    header = {
        "msg_id": uuid.uuid4().hex,
        "session": session_id,
        "msg_type": msg_type,
        "version": "5.3",
    }
    return json.dumps(
        {
            "header": header,
            "parent_header": {},
            "metadata": {},
            "content": content,
            "buffers": [],
            "channel": "shell",
        }
    )


class ColabExecutor:
    """Executes code on a Colab runtime and streams output."""

    def __init__(self, runtime: ColabRuntime):
        if not runtime.info:
            raise ValueError("Runtime must be assigned before creating an executor.")
        self.runtime = runtime
        self._proxy_url = runtime.info.proxy_url
        self._proxy_token = runtime.info.proxy_token
        self._kernel_id: str | None = None
        self._http_session = requests.Session()

    def _api_headers(self) -> dict[str, str]:
        """Headers for Jupyter REST API calls via the runtime proxy."""
        return {
            "Authorization": f"Bearer {self.runtime.credentials.token}",
            "X-Colab-Runtime-Proxy-Token": self._proxy_token,
            "X-Colab-Client-Agent": CLIENT_AGENT_HEADER,
            "Content-Type": "application/json",
        }

    def _create_session(self) -> str:
        """Create a Jupyter session and return the kernel ID."""
        url = f"{self._proxy_url}/api/sessions"
        payload = {
            "kernel": {"name": "python3"},
            "name": "colaber-session",
            "path": "colaber.ipynb",
            "type": "notebook",
        }
        resp = self._http_session.post(
            url, json=payload, headers=self._api_headers()
        )
        resp.raise_for_status()
        data = resp.json()
        self._kernel_id = data["kernel"]["id"]
        return self._kernel_id

    def _delete_session(self, session_id: str) -> None:
        """Clean up the Jupyter session."""
        url = f"{self._proxy_url}/api/sessions/{session_id}"
        try:
            self._http_session.delete(url, headers=self._api_headers())
        except Exception:
            pass

    async def _execute_ws(
        self,
        code: str,
        on_stdout: Callable[[str], None],
        on_stderr: Callable[[str], None],
        on_stdin: Callable[[str, bool], str] | None,
        timeout: float | None,
    ) -> ExecutionResult:
        """Connect to the kernel WebSocket and execute code, streaming output."""
        session_id = uuid.uuid4().hex
        kernel_id = self._create_session()
        allow_stdin = on_stdin is not None

        # Build WebSocket URL
        ws_url = self._proxy_url.replace("https://", "wss://").replace(
            "http://", "ws://"
        )
        ws_url = f"{ws_url}/api/kernels/{kernel_id}/channels?session_id={session_id}"

        headers = {
            "Authorization": f"Bearer {self.runtime.credentials.token}",
            "X-Colab-Runtime-Proxy-Token": self._proxy_token,
        }

        result = ExecutionResult(status="ok")

        async with websockets.connect(
            ws_url,
            additional_headers=headers,
            max_size=2**24,  # 16MB max message size
            ping_interval=30,
        ) as ws:
            # Send execute_request
            msg = _jupyter_message(
                "execute_request",
                {
                    "code": code,
                    "silent": False,
                    "store_history": False,
                    "user_expressions": {},
                    "allow_stdin": allow_stdin,
                    "stop_on_error": True,
                },
                session_id,
            )
            await ws.send(msg)

            # Read messages until execute_reply
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    result.status = "error"
                    result.error_name = "TimeoutError"
                    result.error_value = "Execution timed out"
                    # Send interrupt
                    interrupt_msg = _jupyter_message(
                        "interrupt_request", {}, session_id
                    )
                    await ws.send(interrupt_msg)
                    break

                data = json.loads(raw)
                msg_type = data.get("msg_type") or data.get("header", {}).get(
                    "msg_type"
                )
                content = data.get("content", {})

                if msg_type == "stream":
                    text = content.get("text", "")
                    if content.get("name") == "stderr":
                        on_stderr(text)
                    else:
                        on_stdout(text)

                elif msg_type == "input_request" and on_stdin is not None:
                    prompt = content.get("prompt", "")
                    password = content.get("password", False)
                    # Read input from user in a thread to avoid blocking the event loop
                    value = await asyncio.to_thread(on_stdin, prompt, password)
                    reply = _jupyter_message(
                        "input_reply",
                        {"value": value},
                        session_id,
                    )
                    # input_reply goes on the stdin channel
                    reply_data = json.loads(reply)
                    reply_data["channel"] = "stdin"
                    await ws.send(json.dumps(reply_data))

                elif msg_type == "error":
                    result.status = "error"
                    result.error_name = content.get("ename", "")
                    result.error_value = content.get("evalue", "")
                    result.traceback = content.get("traceback", [])

                elif msg_type == "execute_result":
                    text = content.get("data", {}).get("text/plain", "")
                    if text:
                        on_stdout(text + "\n")

                elif msg_type == "execute_reply":
                    if content.get("status") == "error" and result.status != "error":
                        result.status = "error"
                        result.error_name = content.get("ename", "")
                        result.error_value = content.get("evalue", "")
                        result.traceback = content.get("traceback", [])
                    elif result.status != "error":
                        result.status = content.get("status", "ok")
                    break

        return result

    async def _execute_batch_ws(
        self,
        code_items: list[str],
        on_each_complete: Callable[[int], None] | None,
        timeout: float | None,
    ) -> ExecutionResult:
        """Execute multiple code snippets over a single WebSocket.

        Sends all execute_request messages up front (pipelining), then
        collects execute_reply messages in order.  This eliminates
        per-message round-trip latency.
        """
        session_id = uuid.uuid4().hex
        kernel_id = self._create_session()

        ws_url = self._proxy_url.replace("https://", "wss://").replace(
            "http://", "ws://"
        )
        ws_url = f"{ws_url}/api/kernels/{kernel_id}/channels?session_id={session_id}"

        headers = {
            "Authorization": f"Bearer {self.runtime.credentials.token}",
            "X-Colab-Runtime-Proxy-Token": self._proxy_token,
        }

        result = ExecutionResult(status="ok")

        async with websockets.connect(
            ws_url,
            additional_headers=headers,
            max_size=2**24,
            ping_interval=30,
        ) as ws:
            # Phase 1: Send ALL execute_request messages immediately
            msg_ids: list[str] = []
            for code in code_items:
                msg_data = json.loads(
                    _jupyter_message(
                        "execute_request",
                        {
                            "code": code,
                            "silent": False,
                            "store_history": False,
                            "user_expressions": {},
                            "allow_stdin": False,
                            "stop_on_error": True,
                        },
                        session_id,
                    )
                )
                msg_ids.append(msg_data["header"]["msg_id"])
                await ws.send(json.dumps(msg_data))

            # Phase 2: Collect execute_reply messages in order
            replies_received = 0
            while replies_received < len(code_items):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    result.status = "error"
                    result.error_name = "TimeoutError"
                    result.error_value = "Execution timed out"
                    return result

                data = json.loads(raw)
                msg_type = data.get("msg_type") or data.get("header", {}).get("msg_type")
                content = data.get("content", {})

                if msg_type == "error":
                    result.status = "error"
                    result.error_name = content.get("ename", "")
                    result.error_value = content.get("evalue", "")
                    result.traceback = content.get("traceback", [])
                    return result

                if msg_type == "execute_reply":
                    if content.get("status") == "error":
                        result.status = "error"
                        result.error_name = content.get("ename", "")
                        result.error_value = content.get("evalue", "")
                        result.traceback = content.get("traceback", [])
                        return result

                    if on_each_complete:
                        on_each_complete(replies_received)
                    replies_received += 1

        return result

    def execute(
        self,
        code: str,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
        on_stdin: Callable[[str, bool], str] | None = _default_stdin,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Execute code on the Colab runtime, streaming output.

        Args:
            code: Python code to execute.
            on_stdout: Callback for stdout output. Defaults to sys.stdout.write.
            on_stderr: Callback for stderr output. Defaults to sys.stderr.write.
            on_stdin: Callback for input requests. Receives (prompt, is_password),
                returns the user's input string. Defaults to terminal input.
                Pass None to disable stdin (kernel will error on input()).
            timeout: Max seconds to wait for execution. None means no timeout.

        Returns:
            ExecutionResult with status and any error information.
        """
        if on_stdout is None:
            on_stdout = sys.stdout.write
        if on_stderr is None:
            on_stderr = sys.stderr.write

        return asyncio.run(self._execute_ws(code, on_stdout, on_stderr, on_stdin, timeout))

    def execute_batch(
        self,
        code_items: list[str],
        on_each_complete: Callable[[int], None] | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Execute multiple code snippets over a single WebSocket connection.

        Much faster than calling execute() in a loop, since the WebSocket and
        Jupyter session are created only once.

        Args:
            code_items: List of Python code strings to execute sequentially.
            on_each_complete: Optional callback called after each snippet completes,
                with the snippet index (0-based).
            timeout: Max seconds to wait for each individual execution.

        Returns:
            ExecutionResult — returns immediately on first error.
        """
        return asyncio.run(self._execute_batch_ws(code_items, on_each_complete, timeout))
