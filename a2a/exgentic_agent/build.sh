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
    local tag=$3
    local use_cache=$4
    
    print_info "Building ${image_name}:${tag} using ${runtime}..."
    if [ "$use_cache" = "false" ]; then
        print_info "Building without cache (default)"
    else
        print_info "Building with cache enabled"
    fi
    
    # Build the image with the agent name as build arg
    local build_cmd="$runtime build"
    if [ "$use_cache" = "false" ]; then
        build_cmd="$build_cmd --no-cache"
    fi
    
    if $build_cmd \
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
    AGENT=""
    TAG="latest"
    USE_CACHE="false"  # Default: do not use cache for consistency
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tag)
                TAG="$2"
                shift 2
                ;;
            --use-cache)
                USE_CACHE="true"
                shift
                ;;
            --help|-h)
                cat << EOF
Usage: $0 AGENT_NAME [--tag TAG] [--use-cache]

Build exgentic a2a agent Docker/Podman image.

Arguments:
  AGENT_NAME     Agent name (required, positional)
  --tag TAG      Image tag (optional, default: latest)
  --use-cache    Use Docker cache during build (optional, default: no cache for consistency)

Examples:
  $0 my_agent                      # Build without cache (default)
  $0 my_agent --tag v1.0.0         # Build v1.0.0 without cache
  $0 my_agent --use-cache          # Build with cache enabled
  $0 my_agent --tag v1.0.0 --use-cache  # Build v1.0.0 with cache

The script automatically detects whether to use docker or podman.
By default, builds do not use cache to ensure consistency.
EOF
                exit 0
                ;;
            -*)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
            *)
                if [ -z "$AGENT" ]; then
                    AGENT="$1"
                    shift
                else
                    print_error "Unexpected argument: $1"
                    echo "Use --help for usage information"
                    exit 1
                fi
                ;;
        esac
    done
    
    # Validate agent name is provided
    if [ -z "$AGENT" ]; then
        print_error "Agent name is required!"
        echo ""
        echo "Usage: $0 AGENT_NAME [--tag TAG] [--no-cache]"
        echo ""
        echo "Example: $0 my_agent --tag v1.0.0"
        echo "Use --help for more information"
        exit 1
    fi
    
    AGENTS=("$AGENT")
    
    print_info "Building agent: ${AGENT}"
    print_info "Image tag: ${TAG}"
    if [ "$USE_CACHE" = "true" ]; then
        print_info "Cache: enabled"
    else
        print_info "Cache: disabled (default)"
    fi
    echo ""
    
    # Build each agent
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    
    for agent in "${AGENTS[@]}"; do
        if build_agent "$agent" "$RUNTIME" "$TAG" "$USE_CACHE"; then
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

main "$@"

# Made with Bob