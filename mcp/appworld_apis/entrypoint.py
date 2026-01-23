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


def ensure_appworld_installed() -> None:
    """Ensure appworld is fully installed before any imports that require it."""
    appworld_root = os.environ.get("APPWORLD_ROOT", "/app")
    
    print(f"APPWORLD_ROOT is set to: {appworld_root}")
    
    # Ensure the directory exists
    os.makedirs(appworld_root, exist_ok=True)
    update_root(appworld_root)
    
    # Monkey-patch the verify function BEFORE any imports that use it
    # This prevents the exception from being raised during module imports
    import appworld.cli
    original_verify = appworld.cli.verify_fully_installed
    
    # Track if we've already checked/installed
    installation_checked = False
    
    def patched_verify():
        """Patched version that doesn't raise during imports."""
        nonlocal installation_checked
        if not installation_checked:
            # First time - do the actual check and install if needed
            installation_checked = True
            try:
                original_verify()
                print("✓ appworld is fully installed")
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
                    original_verify()
                except subprocess.CalledProcessError as install_error:
                    print(f"✗ Installation failed: {install_error}")
                    print(f"stdout: {install_error.stdout}")
                    print(f"stderr: {install_error.stderr}")
                    raise
        # After first check, always succeed (don't raise)
    
    # Replace the function globally
    appworld.cli.verify_fully_installed = patched_verify
    
    # Trigger the check now
    patched_verify()


def run_apis() -> None:
    # Import only after ensuring appworld is installed
    from appworld.serve.apis import run

    port = int(os.environ.get("APIS_PORT", "8000"))
    num_workers = int(os.environ.get("APIS_NUM_WORKERS", "1"))
    on_disk = _str_is_true(os.environ.get("APIS_ON_DISK", "1"))
    run(port=port, num_workers=num_workers, on_disk=on_disk)


def run_mcp() -> None:
    # Import only after ensuring appworld is installed
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
