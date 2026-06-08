SHELL := /bin/bash

.PHONY: lint fmt test-docker test-a2a test-mcp test-exgentic-agent test-exgentic-tool sync-all-uv sync-a2a sync-mcp test-startup-all test-startup-a2a test-startup-mcp

lint:
	pre-commit run --all-files

fmt:
	ruff format .
	ruff check --fix .

# This builds all of the A2A and MCP example Docker images to verify they can be built successfully.
test-docker: test-a2a test-mcp test-exgentic-agent test-exgentic-tool

# Directories under a2a/ to skip in test-a2a
TEST_A2A_SKIP := a2a/exgentic_agent

# Verify all of the A2A example Docker images can be built
# (Optional KAGENTI_DOCKER_FLAGS for docker build, e.g. --no-cache or --load)
test-a2a:
	@for f in $(shell find a2a -mindepth 1 -maxdepth 1 -type d); do \
		case " $(TEST_A2A_SKIP) " in \
			*" $${f} "*) echo "Skipping $${f}..."; continue;; \
		esac; \
		pushd $${f} || exit; \
		echo "Building Docker image for $${f}..."; \
		docker build ${KAGENTI_DOCKER_FLAGS} --tag $${f##*/} . || exit; \
		popd; \
	done

# Directories under mcp/ to skip in test-mcp
TEST_MCP_SKIP := mcp/exgentic_benchmarks mcp/wiki_memory_tool

# Verify all of the MCP example Docker images can be built
# (Optional KAGENTI_DOCKER_FLAGS for docker build, e.g. --no-cache or --load)
test-mcp:
	@for f in $(shell find mcp -mindepth 1 -maxdepth 1 -type d); do \
		case " $(TEST_MCP_SKIP) " in \
			*" $${f} "*) echo "Skipping $${f}..."; continue;; \
		esac; \
		pushd $${f} || exit; \
		echo "Building Docker image for $${f}..."; \
		docker build ${KAGENTI_DOCKER_FLAGS} --tag $${f##*/} . || exit; \
		popd; \
	done

# Build the exgentic agent (skipped from test-a2a)
test-exgentic-agent:
	./a2a/exgentic_agent/build.sh tool_calling

# Build the exgentic benchmarks tool (skipped from test-mcp)
test-exgentic-tool:
	./mcp/exgentic_benchmarks/build.sh tau2

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

test-startup-all: test-startup-a2a test-startup-mcp

# Run the test_startup.exp script for each A2A example that has one to verify it starts successfully.
test-startup-a2a:
	@for f in $(shell find a2a -mindepth 1 -maxdepth 1 -type d); do \
		pushd $${f} || exit; \
		if [ -f test_startup.exp ]; then \
			echo "Testing startup for $${f}..."; \
			expect -f test_startup.exp || exit; \
		fi; \
		popd; \
	done

test-startup-mcp:
	@for f in $(shell find mcp -mindepth 1 -maxdepth 1 -type d); do \
		pushd $${f} || exit; \
		if [ -f test_startup.exp ]; then \
			echo "Testing startup for $${f}..."; \
			expect -f test_startup.exp || exit; \
		fi; \
		popd; \
	done
