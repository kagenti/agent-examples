"""Module entry point so the agent can be started with ``uv run .``.

The server logic lives in :mod:`server`, which is also exposed as the ``server``
console script via ``[project.scripts]``.
"""

from server import main

if __name__ == "__main__":
    main()
