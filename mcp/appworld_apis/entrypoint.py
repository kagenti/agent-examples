import os
import signal
import sys
from multiprocessing import Process

from appworld import update_root


def _str_is_true(value: str | None) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


def run_apis() -> None:
    from appworld.serve.apis import run

    port = int(os.environ.get("APIS_PORT", "8000"))
    num_workers = int(os.environ.get("APIS_NUM_WORKERS", "1"))
    on_disk = _str_is_true(os.environ.get("APIS_ON_DISK", "1"))
    run(port=port, num_workers=num_workers, on_disk=on_disk)


def run_mcp() -> None:
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
    appworld_root = os.environ.get("APPWORLD_ROOT")
    if appworld_root:
        update_root(appworld_root)
    print(appworld_root, os.environ.get("APPWORLD_CACHE"))

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
