import asyncio
import json
import logging
import os

from claude_agent.configuration import Configuration
from claude_agent.events import StreamTranslator
from claude_agent.session import ClaudeSession

logger = logging.getLogger(__name__)


def build_argv(session: ClaudeSession, prompt: str, model: str | None) -> list[str]:
    argv = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
    ]
    if session.started:
        argv += ["--resume", session.session_uuid]
    else:
        argv += ["--session-id", session.session_uuid]
    if model:
        argv += ["--model", model]
    return argv


def build_env(config: Configuration) -> dict[str, str]:
    # NOTE: this copies the full pod environment so the `claude` subprocess has a
    # working runtime (PATH, HOME, TLS/CA, Node vars, etc.). Because the agent runs
    # Claude with --dangerously-skip-permissions, a prompt can execute arbitrary
    # code inside this container and read these env vars. Trust model: prompts are
    # fully trusted and the container/pod is the isolation boundary — do not
    # co-locate unrelated secrets in this pod, and isolate per tenant.
    env = dict(os.environ)
    if config.anthropic_base_url:
        env["ANTHROPIC_BASE_URL"] = config.anthropic_base_url
    env["ANTHROPIC_AUTH_TOKEN"] = config.anthropic_auth_token
    env["ANTHROPIC_MODEL"] = config.anthropic_model
    if config.anthropic_default_haiku_model:
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = config.anthropic_default_haiku_model
    env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = config.disable_experimental_betas
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = config.disable_nonessential_traffic
    # Writable HOME so the subprocess can read/write ~/.claude.
    env["HOME"] = config.home_dir
    # Never let the subprocess inherit an x-api-key path; we use the bearer token.
    env.pop("ANTHROPIC_API_KEY", None)
    return env


async def _consume(stdout, translator: StreamTranslator) -> None:
    async for raw in stdout:
        line = raw.decode(errors="replace").strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("skipping non-JSON line: %s", line[:200])
            continue
        await translator.handle(event)


async def _drain(stream, sink: list[bytes]) -> None:
    async for chunk in stream:
        sink.append(chunk)


async def run_turn(
    session: ClaudeSession,
    prompt: str,
    translator: StreamTranslator,
    config: Configuration,
) -> None:
    """Run one conversational turn as a `claude` subprocess in the session's cwd.

    Caller must hold `session.lock`. On any failure, sets the translator's error
    state; the caller calls `translator.finish()` to emit the terminal A2A state.
    """
    workdir = session.ensure_workdir()
    os.makedirs(config.home_dir, exist_ok=True)
    argv = build_argv(session, prompt, config.anthropic_model)
    env = build_env(config)

    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=workdir,
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stderr_chunks: list[bytes] = []
    stderr_task = asyncio.ensure_future(_drain(proc.stderr, stderr_chunks))

    try:
        try:
            await asyncio.wait_for(_consume(proc.stdout, translator), timeout=config.turn_timeout_s)
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            translator.errored = True
            translator.error_reason = f"turn timed out after {config.turn_timeout_s}s"
            return

        await stderr_task

        if proc.returncode != 0 and translator.final_text is None:
            stderr = b"".join(stderr_chunks).decode(errors="replace")
            logger.error("claude exited %s: %s", proc.returncode, stderr[:500])
            translator.errored = True
            translator.error_reason = f"claude exited with code {proc.returncode}"
        else:
            session.started = True
    finally:
        # Never let the subprocess outlive this turn — on timeout, error, or
        # cancellation (A2A cancel / client disconnect). A lingering
        # --dangerously-skip-permissions process is especially undesirable.
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        if not stderr_task.done():
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass
