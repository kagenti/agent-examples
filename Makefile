.PHONY: lint fmt test-docker test-a2a test-mcp sync-all-uv sync-a2a sync-mcp

lint:
	pre-commit run --all-files

fmt:
	ruff format .
	ruff check --fix .

# This builds all of the A2A and MCP example Docker images to verify they can be built successfully.
test-docker: test-a2a test-mcp

# Verify all of the A2A example Docker images can be built
# (Optional KAGENTI_DOCKER_FLAGS for docker build, e.g. --no-cache or --load)
test-a2a:
	@for f in $(shell find a2a -mindepth 1 -maxdepth 1 -type d); do \
		pushd $${f} || exit; \
		echo "Building Docker image for $${f}..."; \
		docker build ${KAGENTI_DOCKER_FLAGS} --tag $${f##*/} . || exit; \
		popd; \
	done

# Verify all of the MCP example Docker images can be built
# (Optional KAGENTI_DOCKER_FLAGS for docker build, e.g. --no-cache or --load)
test-mcp:
	@for f in $(shell find mcp -mindepth 1 -maxdepth 1 -type d); do \
		pushd $${f} || exit; \
		echo "Building Docker image for $${f}..."; \
		docker build ${KAGENTI_DOCKER_FLAGS} --tag $${f##*/} . || exit; \
		popd; \
	done

# After changing pyproject.toml, run this to sync dependencies for all examples
sync-all-uv: sync-a2a sync-mcp

sync-a2a:
	@for f in $(shell find a2a -mindepth 1 -maxdepth 1 -type d); do \
		pushd $${f} || exit; \
		echo "Syncing dependencies for $${f}..."; \
		uv sync --no-dev || exit; \
		popd; \
	done

sync-mcp:
	@for f in $(shell find mcp -mindepth 1 -maxdepth 1 -type d); do \
		pushd $${f} || exit; \
		echo "Syncing dependencies for $${f}..."; \
		uv sync --no-dev || exit; \
		popd; \
	done
