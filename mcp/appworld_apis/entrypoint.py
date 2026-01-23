import os
import signal
import subprocess
import sys
from multiprocessing import Process

from appworld import update_root


def _str_is_true(value: str | None) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


def patch_appworld_verify() -> None:
    """Monkey-patch appworld's verify function to not raise exceptions."""
    import appworld.cli
    
    # Save the original function
    if not hasattr(appworld.cli, '_original_verify_fully_installed'):
        appworld.cli._original_verify_fully_installed = appworld.cli.verify_fully_installed
    
    def patched_verify():
        """Patched version that never raises - just logs."""
        try:
            appworld.cli._original_verify_fully_installed()
            print("✓ appworld is fully installed")
        except Exception as e:
            print(f"⚠ appworld verification skipped (will be handled at startup): {e}")
    
    # Replace the function
    appworld.cli.verify_fully_installed = patched_verify


def ensure_appworld_installed() -> None:
    """Ensure appworld is fully installed before any imports that require it."""
    appworld_root = os.environ.get("APPWORLD_ROOT", "/app")
    
    print(f"APPWORLD_ROOT is set to: {appworld_root}")
    
    # Ensure the directory exists
    os.makedirs(appworld_root, exist_ok=True)
    update_root(appworld_root)
    
    # First, patch the verify function so imports don't fail
    patch_appworld_verify()
    
    # Now do the actual installation check using the original function
    import appworld.cli
    try:
        appworld.cli._original_verify_fully_installed()
        print("✓ appworld is already fully installed")
    except Exception as e:
        print(f"✗ appworld not fully installed: {e}")
        print("Running appworld install...")
        try:
            result = subprocess.run(
                ["appworld", "install"],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "APPWORLD_ROOT": appworld_root}
            )
            print(result.stdout)
            print("✓ appworld install completed successfully")
            # Verify it worked
            appworld.cli._original_verify_fully_installed()
        except subprocess.CalledProcessError as install_error:
            print(f"✗ Installation failed: {install_error}")
            print(f"stdout: {install_error.stdout}")
            print(f"stderr: {install_error.stderr}")
            raise


def run_apis() -> None:
    # Each child process needs to ensure appworld is installed
    # and apply the monkey-patch before importing
    ensure_appworld_installed()
    
    from appworld.serve.apis import run

    port = int(os.environ.get("APIS_PORT", "8000"))
    num_workers = int(os.environ.get("APIS_NUM_WORKERS", "1"))
    on_disk = _str_is_true(os.environ.get("APIS_ON_DISK", "1"))
    run(port=port, num_workers=num_workers, on_disk=on_disk)


def run_mcp() -> None:
    # Each child process needs to ensure appworld is installed
    # and apply the monkey-patch before importing
    ensure_appworld_installed()
    
    from appworld.serve import _mcp

    #transport = os.environ.get("MCP_TRANSPORT", "http")
    transport = "http"
    port = int(os.environ.get("MCP_PORT", "8001"))
    remote_apis_url = os.environ.get("REMOTE_APIS_URL", "http://localhost:8000")
    output_type = os.environ.get("MCP_OUTPUT_TYPE", "both")
    app_names_raw = os.environ.get("MCP_APP_NAMES")
    app_names = None
    if app_names_raw:
        app_names = tuple(name for name in app_names_raw.split(",") if name)
    _mcp.run(
        transport=transport,
        app_names=app_names,
        output_type=output_type,
        remote_apis_url=remote_apis_url,
        port=port,
    )


def main() -> None:
    # Ensure appworld is installed BEFORE starting any processes
    ensure_appworld_installed()

    apis_process = Process(target=run_apis)
    mcp_process = Process(target=run_mcp)

    apis_process.start()
    mcp_process.start()

    def shutdown(signum: int, _frame: object | None) -> None:
        for process in (apis_process, mcp_process):
            if process.is_alive():
                process.terminate()
        for process in (apis_process, mcp_process):
            process.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    exit_code = 0
    try:
        while True:
            if not apis_process.is_alive():
                exit_code = apis_process.exitcode or 1
                break
            if not mcp_process.is_alive():
                exit_code = mcp_process.exitcode or 1
                break
            signal.pause()
    finally:
        shutdown(signal.SIGTERM, None)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
