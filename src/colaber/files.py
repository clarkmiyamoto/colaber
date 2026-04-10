"""File transfer between local machine and Colab runtime."""

from __future__ import annotations

import base64
import io
import os
import sys
import tarfile
import time
from pathlib import Path

import pathspec
import requests

from colaber.config import CLIENT_AGENT_HEADER, REMOTE_PROJECT_DIR, UPLOAD_EXCLUDE_DIRS
from colaber.executor import ColabExecutor
from colaber.runtime import ColabRuntime


def _format_size(n: float) -> str:
    """Format bytes as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"




def _load_gitignore(project_dir: Path) -> pathspec.PathSpec | None:
    """Load .gitignore patterns from the project directory."""
    gitignore_path = project_dir / ".gitignore"
    if not gitignore_path.exists():
        return None
    return pathspec.PathSpec.from_lines("gitwildmatch", gitignore_path.read_text().splitlines())


def _should_include(path: Path, project_dir: Path, gitignore: pathspec.PathSpec | None) -> bool:
    """Check if a file should be included in the upload archive."""
    rel = path.relative_to(project_dir)

    # Skip excluded directories
    for part in rel.parts:
        if part in UPLOAD_EXCLUDE_DIRS:
            return False

    # Skip gitignore matches
    if gitignore and gitignore.match_file(str(rel)):
        return False

    return True


def create_project_archive(project_dir: Path) -> bytes:
    """Create a tar.gz archive of the project directory.

    Respects .gitignore and excludes common non-essential directories.
    Returns the archive as bytes.
    """
    project_dir = project_dir.resolve()
    gitignore = _load_gitignore(project_dir)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, dirs, files in os.walk(project_dir):
            root_path = Path(root)

            # Prune excluded directories in-place to avoid descending
            dirs[:] = [
                d for d in dirs
                if d not in UPLOAD_EXCLUDE_DIRS
                and _should_include(root_path / d, project_dir, gitignore)
            ]

            for fname in files:
                fpath = root_path / fname
                if not _should_include(fpath, project_dir, gitignore):
                    continue
                arcname = str(fpath.relative_to(project_dir))
                tar.add(str(fpath), arcname=arcname)

    return buf.getvalue()


def _contents_api_headers(runtime: ColabRuntime) -> dict[str, str]:
    """Build headers for Jupyter Contents API requests."""
    return {
        "Authorization": f"Bearer {runtime.credentials.token}",
        "X-Colab-Runtime-Proxy-Token": runtime.info.proxy_token,
        "X-Colab-Client-Agent": CLIENT_AGENT_HEADER,
    }


def _render_progress(sent: int, total: int, start: float) -> None:
    """Render a progress bar to stderr."""
    elapsed = time.monotonic() - start
    speed = sent / elapsed if elapsed > 0 else 0
    pct = sent / total * 100 if total else 100
    bar_width = 30
    filled = int(bar_width * pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    line = (
        f"\r  {bar} {pct:5.1f}%  "
        f"{_format_size(sent)}/{_format_size(total)}  "
        f"{_format_size(speed)}/s"
    )
    sys.stderr.write(line)
    sys.stderr.flush()


def _upload_via_kernel(
    executor: ColabExecutor,
    archive: bytes,
) -> None:
    """Upload archive via Jupyter kernel execute_request."""
    archive_b64 = base64.b64encode(archive).decode("ascii")
    total_size = len(archive)

    code_items: list[str] = []

    code_items.append(
        f'import os\nos.makedirs("{REMOTE_PROJECT_DIR}", exist_ok=True)\n'
        f'_colaber_f = open("/content/_colaber_upload.b64", "w")\n'
    )

    chunk_size = 8_000_000
    for i in range(0, len(archive_b64), chunk_size):
        chunk = archive_b64[i : i + chunk_size]
        code_items.append(f'_colaber_f.write("{chunk}")\n')

    code_items.append(
        f"import base64, tarfile\n"
        f"_colaber_f.close()\n"
        f"del _colaber_f\n"
        f'with open("/content/_colaber_upload.b64", "r") as f:\n'
        f"    raw = base64.b64decode(f.read())\n"
        f'with open("/content/_colaber_upload.tar.gz", "wb") as f:\n'
        f"    f.write(raw)\n"
        f"del raw\n"
        f"import os\n"
        f'os.remove("/content/_colaber_upload.b64")\n'
        f'with tarfile.open("/content/_colaber_upload.tar.gz", "r:gz") as tf:\n'
        f'    tf.extractall("{REMOTE_PROJECT_DIR}")\n'
        f'os.remove("/content/_colaber_upload.tar.gz")\n'
        f'os.chdir("{REMOTE_PROJECT_DIR}")\n'
    )

    total_b64 = len(archive_b64)
    total_items = len(code_items)
    start = time.monotonic()

    def on_progress(idx: int) -> None:
        if idx == 0 or idx == total_items - 1:
            return
        chunks_done = idx
        sent_b64 = min(chunks_done * chunk_size, total_b64)
        sent_raw = int(sent_b64 / total_b64 * total_size) if total_b64 else total_size
        _render_progress(sent_raw, total_size, start)

    result = executor.execute_batch(code_items, on_each_complete=on_progress)

    sys.stderr.write("\n")
    sys.stderr.flush()

    if result.status != "ok":
        raise RuntimeError(f"Failed to upload/extract project: {result.error_value}")


def upload_project(runtime: ColabRuntime, executor: ColabExecutor, project_dir: Path) -> None:
    """Upload the project directory to the Colab runtime.

    Creates a tar.gz archive, sends it via pipelined kernel execute_request
    messages over a single WebSocket connection, then extracts it.
    """
    archive = create_project_archive(project_dir)
    _upload_via_kernel(executor, archive)


def snapshot_remote_files(executor: ColabExecutor) -> set[str]:
    """Get a snapshot of files in the remote project directory."""
    files: list[str] = []

    def capture(text: str) -> None:
        for line in text.strip().splitlines():
            if line.strip():
                files.append(line.strip())

    code = f"""
import os
for root, dirs, fnames in os.walk("{REMOTE_PROJECT_DIR}"):
    for f in fnames:
        print(os.path.join(root, f))
"""
    executor.execute(code, on_stdout=capture, on_stderr=lambda _: None, on_stdin=None)
    return set(files)


def download_outputs(
    runtime: ColabRuntime,
    new_files: set[str],
    output_dir: Path,
) -> list[Path]:
    """Download newly created files from the Colab runtime.

    Args:
        runtime: The assigned Colab runtime.
        new_files: Set of remote file paths to download.
        output_dir: Local directory to save files to.

    Returns:
        List of local paths where files were saved.
    """
    headers = _contents_api_headers(runtime)

    downloaded = []
    for remote_path in sorted(new_files):
        # Convert absolute path to Contents API path (relative to /content/)
        contents_path = remote_path.replace("/content/", "", 1)
        url = f"{runtime.info.proxy_url}/api/contents/{contents_path}?content=1"

        try:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            content_b64 = data.get("content", "")
            file_bytes = base64.b64decode(content_b64)

            # Determine local path (strip the project/ prefix)
            rel_path = remote_path.replace(f"{REMOTE_PROJECT_DIR}/", "", 1)
            local_path = output_dir / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(file_bytes)
            downloaded.append(local_path)
        except Exception:
            pass  # Skip files that fail to download

    return downloaded
