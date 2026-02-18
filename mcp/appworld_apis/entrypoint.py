import os
import signal
import sys
import threading
from inspect import signature
from importlib import import_module
from multiprocessing import Process

from appworld import update_root


def _str_is_true(value: str | None) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


def _ensure_under(base_dir: str, path: str) -> bool:
    base_dir = os.path.abspath(base_dir)
    path = os.path.abspath(path)
    return path == base_dir or path.startswith(base_dir + os.sep)


def _coerce_db_path_for_docker_mode(path: str | None, appworld_root: str) -> str | None:
    if path is None:
        return None
    if path.lower().startswith(":memory:"):
        return path

    data_root = os.path.join(appworld_root, "data")
    outputs_root = os.path.join(appworld_root, "experiments", "outputs")
    if os.path.isabs(path):
        normalized = path.replace("\\", "/")
        data_marker = "/data/"
        outputs_marker = "/experiments/outputs/"
        if data_marker in normalized:
            suffix = normalized.split(data_marker, 1)[1]
            path = os.path.join(appworld_root, "data", suffix)
        elif outputs_marker in normalized:
            suffix = normalized.split(outputs_marker, 1)[1]
            path = os.path.join(appworld_root, "experiments", "outputs", suffix)

    resolved = os.path.abspath(path if os.path.isabs(path) else os.path.join(appworld_root, path))
    if _ensure_under(data_root, resolved) or _ensure_under(outputs_root, resolved):
        return resolved
    raise ValueError(
        "DB path is outside allowed roots. "
        "Allowed: APPWORLD_ROOT/data and APPWORLD_ROOT/experiments/outputs"
    )


def _enable_docker_mode_db_guard() -> None:
    appworld_root = os.path.abspath(os.environ.get("APPWORLD_ROOT", os.getcwd()))
    default_disk_base = os.path.join(appworld_root, "data", "base_dbs")
    for module_name in (
        "appworld.apps.model_lib",
        "appworld.model_lib",
        "appworld.apps",
    ):
        try:
            candidate = import_module(module_name)
        except ModuleNotFoundError:
            continue
        get_db_home_path = getattr(candidate, "get_db_home_path", None)
        if callable(get_db_home_path):
            try:
                default_disk_base = get_db_home_path(storage_type="disk", type="base")
            except Exception:
                pass
            break
    api_module = None
    for module_name in (
        "appworld.apps.lib.apis.local_remote",
        "appworld.apps.api_lib",
        "appworld.api_lib",
        "appworld.apps",
    ):
        try:
            candidate = import_module(module_name)
        except ModuleNotFoundError:
            continue
        if hasattr(candidate, "set_local_dbs"):
            api_module = candidate
            break

    if api_module is None:
        print("APIS_DOCKER_MODE enabled, but no set_local_dbs module was found; skipping guard.")
        return

    original_set_local_dbs = getattr(api_module, "set_local_dbs")
    original_save_local_dbs = getattr(api_module, "save_local_dbs", None)
    raise_http_exception = getattr(api_module, "raise_http_exception", None)

    def guarded_set_local_dbs(
        to_db_home_path: str | None = None,
        from_db_home_path: str | None = None,
        app_names: list[str] | None = None,
        create: bool = False,
    ) -> None:
        # Coerce paths into allowed roots; fall back to the default disk base
        # when a path can't be mapped (e.g. AppWorld's own default that lives
        # outside the container's APPWORLD_ROOT).
        try:
            to_db_home_path = _coerce_db_path_for_docker_mode(to_db_home_path, appworld_root)
        except ValueError:
            print(f"[docker-mode] to_db_home_path {to_db_home_path!r} outside allowed roots, remapping to default")
            to_db_home_path = None
        try:
            from_db_home_path = _coerce_db_path_for_docker_mode(from_db_home_path, appworld_root)
        except ValueError:
            print(f"[docker-mode] from_db_home_path {from_db_home_path!r} outside allowed roots, remapping to default")
            from_db_home_path = None

        if to_db_home_path is None:
            to_db_home_path = default_disk_base
        if from_db_home_path is None:
            from_db_home_path = default_disk_base

        try:
            original_set_local_dbs(
                to_db_home_path=to_db_home_path,
                from_db_home_path=from_db_home_path,
                app_names=app_names,
                create=create,
            )
        except TypeError:
            original_set_local_dbs(to_db_home_path, from_db_home_path)

    setattr(api_module, "set_local_dbs", guarded_set_local_dbs)

    if callable(original_save_local_dbs):
        save_sig = signature(original_save_local_dbs)

        def guarded_save_local_dbs(*args, **kwargs) -> None:
            try:
                bound = save_sig.bind_partial(*args, **kwargs)
                arguments = dict(bound.arguments)
                arguments["to_db_home_path"] = _coerce_db_path_for_docker_mode(
                    arguments.get("to_db_home_path"),
                    appworld_root,
                )
                arguments["from_db_home_path"] = _coerce_db_path_for_docker_mode(
                    arguments.get("from_db_home_path"),
                    appworld_root,
                )
            except ValueError as error:
                if callable(raise_http_exception):
                    raise_http_exception(str(error), status_code=422)
                raise

            original_save_local_dbs(**arguments)

        setattr(api_module, "save_local_dbs", guarded_save_local_dbs)


def run_apis() -> None:
    from appworld.serve.apis import run

    port = int(os.environ.get("APIS_PORT", "8000"))
    docker_mode = _str_is_true(os.environ.get("APIS_DOCKER_MODE", "1"))
    if docker_mode:
        _enable_docker_mode_db_guard()
    run(port=port)


def run_mcp() -> None:
    from appworld.serve import _mcp

    docker_mode = _str_is_true(os.environ.get("APIS_DOCKER_MODE", "1"))
    if docker_mode:
        _enable_docker_mode_db_guard()

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

    apis_process = Process(target=run_apis)
    mcp_process = Process(target=run_mcp)

    apis_process.start()
    mcp_process.start()

    def shutdown() -> None:
        """Terminate child processes gracefully, then forcefully if needed."""
        for process in (apis_process, mcp_process):
            if process.is_alive():
                process.terminate()
        for process in (apis_process, mcp_process):
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)

    # Event is set by the signal handler to wake the loop immediately on
    # SIGTERM/SIGINT; otherwise we poll child processes every 5 seconds.
    _shutdown_event = threading.Event()

    def _signal_handler(signum: int, _frame: object | None) -> None:
        _shutdown_event.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    exit_code = 0
    try:
        while True:
            if not apis_process.is_alive():
                exit_code = apis_process.exitcode or 1
                break
            if not mcp_process.is_alive():
                exit_code = mcp_process.exitcode or 1
                break
            _shutdown_event.wait(timeout=5)
            if _shutdown_event.is_set():
                break
    finally:
        shutdown()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
