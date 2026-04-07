#!/bin/bash
set -e

# Script to build exgentic a2a agent images
# Automatically detects whether to use docker or podman

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect container runtime (docker or podman)
detect_runtime() {
    if command -v docker &> /dev/null; then
        echo "docker"
    elif command -v podman &> /dev/null; then
        echo "podman"
    else
        print_error "Neither docker nor podman is installed!"
        print_error "Please install one of them to continue."
        exit 1
    fi
}

# Build image for a specific agent
build_agent() {
    local agent=$1
    local runtime=$2
    local image_name="exgentic-a2a-${agent}"
    local tag="${3:-latest}"
    
    print_info "Building ${image_name}:${tag} using ${runtime}..."
    
    # Build the image with the agent name as build arg
    if $runtime build \
        --build-arg AGENT_NAME="${agent}" \
        -t "${image_name}:${tag}" \
        -f Dockerfile \
        .; then
        print_info "✓ Successfully built ${image_name}:${tag}"
        return 0
    else
        print_error "✗ Failed to build ${image_name}:${tag}"
        return 1
    fi
}

# Main script
main() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$script_dir"
    
    print_info "Exgentic A2A Agent Image Builder"
    print_info "================================="
    
    # Detect container runtime
    RUNTIME=$(detect_runtime)
    print_info "Detected container runtime: ${RUNTIME}"
    
    # Parse command line arguments
    AGENT="${1}"
    TAG="${2:-latest}"
    
    # Validate agent name is provided
    if [ -z "$AGENT" ]; then
        print_error "Agent name is required!"
        echo ""
        echo "Usage: $0 AGENT_NAME [TAG]"
        echo ""
        echo "Example: $0 my_agent v1.0.0"
        exit 1
    fi
    
    AGENTS=("$AGENT")
    
    print_info "Building agent: ${AGENT}"
    print_info "Image tag: ${TAG}"
    echo ""
    
    # Build each agent
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    
    for agent in "${AGENTS[@]}"; do
        if build_agent "$agent" "$RUNTIME" "$TAG"; then
            ((SUCCESS_COUNT++))
        else
            ((FAIL_COUNT++))
        fi
        echo ""
    done
    
    # Summary
    print_info "Build Summary"
    print_info "============="
    print_info "Successful: ${SUCCESS_COUNT}"
    if [ $FAIL_COUNT -gt 0 ]; then
        print_error "Failed: ${FAIL_COUNT}"
        exit 1
    else
        print_info "All builds completed successfully!"
        echo ""
        print_info "Built images:"
        for agent in "${AGENTS[@]}"; do
            echo "  - exgentic-a2a-${agent}:${TAG}"
        done
    fi
}

# Show usage if --help is passed
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    cat << EOF
Usage: $0 AGENT_NAME [TAG]

Build exgentic a2a agent Docker/Podman image.

Arguments:
  AGENT_NAME  Agent name (required)
  TAG         Image tag (optional, default: latest)

Examples:
  $0 my_agent              # Build my_agent with 'latest' tag
  $0 my_agent v1.0.0       # Build my_agent with 'v1.0.0' tag
  $0 my_agent dev          # Build my_agent with 'dev' tag

The script automatically detects whether to use docker or podman.
EOF
    exit 0
fi

main "$@"

# Made with Bob