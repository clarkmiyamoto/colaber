"""CLI entry point for colaber."""

from __future__ import annotations

import re
import shlex
import sys
import time
from pathlib import Path

import click

from colaber import __version__


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text (used in Jupyter tracebacks)."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _get_wandb_api_key() -> str | None:
    """Read the WandB API key from the local machine.

    Checks in order:
    1. WANDB_API_KEY environment variable
    2. ~/.netrc (where `wandb login` stores it)
    """
    import os

    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key

    # wandb login stores credentials in ~/.netrc
    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        import netrc

        try:
            info = netrc.netrc(str(netrc_path))
            auth = info.authenticators("api.wandb.ai")
            if auth:
                return auth[2]  # (login, account, password) — password is the key
        except Exception:
            pass

    return None


@click.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("script", type=click.Path(exists=True))
@click.argument("script_args", nargs=-1, type=click.UNPROCESSED)
@click.option(
    "--gpu",
    type=click.Choice(["t4", "l4", "a100"], case_sensitive=False),
    default="t4",
    help="GPU type to use (default: t4).",
)
@click.option("--tpu", is_flag=True, default=False, help="Use TPU instead of GPU.")
@click.option(
    "--requirements",
    type=click.Path(exists=True),
    default=None,
    help="Path to requirements.txt to install before execution.",
)
@click.option(
    "--timeout",
    type=int,
    default=None,
    help="Max runtime in seconds (default: no timeout).",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=".",
    help="Directory to save output files (default: current directory).",
)
@click.option("--wandb", is_flag=True, help="Forward local WandB API key to Colab runtime.")
@click.option("--no-upload", is_flag=True, help="Don't upload project files.")
@click.option("--no-download", is_flag=True, help="Don't download output files.")
@click.version_option(version=__version__)
def main(
    script: str,
    script_args: tuple[str, ...],
    gpu: str,
    tpu: bool,
    requirements: str | None,
    timeout: int | None,
    output_dir: str,
    wandb: bool,
    no_upload: bool,
    no_download: bool,
) -> None:
    """Run a local Python script on Google Colab GPU/TPU.

    Example: colaber main.py --batch-size 32
    """
    from requests.exceptions import HTTPError

    from colaber.auth import clear_cached_credentials, get_credentials
    from colaber.executor import ColabExecutor
    from colaber.files import download_outputs, snapshot_remote_files, upload_project
    from colaber.runtime import ColabRuntime, ColabRuntimeError

    script_path = Path(script).resolve()
    project_dir = script_path.parent
    output_path = Path(output_dir).resolve()
    accelerator = "tpu" if tpu else "gpu"
    start_time = time.time()

    # Step 1: Authenticate
    click.echo("Authenticating with Google...")
    try:
        credentials = get_credentials()
    except Exception as e:
        click.echo(f"Authentication failed: {e}", err=True)
        sys.exit(2)
    click.echo("Authenticated.")

    # Step 2: Assign runtime (with auto-retry on stale token)
    accel_label = "TPU" if tpu else f"GPU ({gpu.upper()})"
    click.echo(f"Requesting Colab runtime ({accel_label})...")

    def _assign_runtime() -> ColabRuntime:
        """Assign a runtime, auto-retrying once on 401 (stale token)."""
        nonlocal credentials
        rt = ColabRuntime(credentials, accelerator=accelerator, gpu_type=gpu)
        try:
            rt.assign()
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                click.echo("Token expired, re-authenticating...")
                clear_cached_credentials()
                credentials = get_credentials()
                click.echo("Authenticated.")
                rt = ColabRuntime(credentials, accelerator=accelerator, gpu_type=gpu)
                rt.assign()
            else:
                raise
        rt._start_keepalive()
        return rt

    runtime = None
    try:
        runtime = _assign_runtime()
        click.echo(f"Runtime assigned: {runtime.info.endpoint}")
        executor = ColabExecutor(runtime)

        # Step 3: Upload project files
        if not no_upload:
            click.echo("Uploading project files...")
            upload_project(runtime, executor, project_dir)
            click.echo("Upload complete.")

        # Step 4: Forward WandB API key
        if wandb:
            wandb_key = _get_wandb_api_key()
            if wandb_key:
                executor.execute(
                    f'import os; os.environ["WANDB_API_KEY"] = "{wandb_key}"',
                    on_stdout=lambda _: None,
                    on_stderr=lambda _: None,
                    on_stdin=None,
                )
                click.echo("WandB API key forwarded.")
            else:
                click.echo(
                    "Warning: --wandb specified but no API key found. "
                    "Run `wandb login` locally first.",
                    err=True,
                )

        # Step 5: Install requirements
        if requirements:
            click.echo("Installing requirements...")
            req_path = Path(requirements).resolve()
            # The requirements.txt is already uploaded as part of the project
            rel_req = req_path.relative_to(project_dir)
            result = executor.execute(
                f"import subprocess; subprocess.run(['pip', 'install', '-r', '{rel_req}'], check=True)",
            )
            if result.status != "ok":
                click.echo(f"Failed to install requirements: {result.error_value}", err=True)
                sys.exit(2)
            click.echo("Requirements installed.")

        # Step 6: Snapshot files before execution (for output detection)
        pre_files = set()
        if not no_download:
            pre_files = snapshot_remote_files(executor)

        # Step 7: Execute script
        script_name = script_path.name
        args_str = " ".join(shlex.quote(a) for a in script_args)
        click.echo(f"Running: {script_name} {args_str}")
        click.echo("---")

        code = f"""\
import sys, runpy
sys.argv = {[script_name] + list(script_args)!r}
runpy.run_path({script_name!r}, run_name='__main__')
"""
        result = executor.execute(code, timeout=timeout)

        click.echo("---")

        # Step 8: Download output files
        downloaded = []
        if not no_download:
            post_files = snapshot_remote_files(executor)
            new_files = post_files - pre_files
            if new_files:
                click.echo(f"Downloading {len(new_files)} new file(s)...")
                downloaded = download_outputs(runtime, new_files, output_path)
                for f in downloaded:
                    click.echo(f"  -> {f}")

        # Summary
        elapsed = time.time() - start_time
        click.echo()
        if result.status == "ok":
            click.echo(f"Completed successfully in {elapsed:.1f}s")
            if downloaded:
                click.echo(f"Downloaded {len(downloaded)} file(s) to {output_path}")
        else:
            click.echo(f"Script failed after {elapsed:.1f}s", err=True)
            if result.traceback:
                click.echo("\nTraceback:", err=True)
                for line in result.traceback:
                    click.echo(_strip_ansi(line), err=True)
            elif result.error_value:
                click.echo(f"{result.error_name}: {result.error_value}", err=True)
            sys.exit(1)

    except ColabRuntimeError as e:
        click.echo(f"Runtime error: {e}", err=True)
        sys.exit(2)
    except KeyboardInterrupt:
        click.echo("\nInterrupted. Releasing runtime...")
        sys.exit(130)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    finally:
        if runtime is not None:
            runtime._stop_keepalive()
            runtime.unassign()
            runtime._session.close()
