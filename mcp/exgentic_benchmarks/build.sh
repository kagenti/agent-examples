#!/bin/bash
set -e

# Script to build exgentic benchmark images for tau2 and gsm8k
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

# Build image for a specific benchmark
build_benchmark() {
    local benchmark=$1
    local runtime=$2
    local image_name="exgentic-mcp-${benchmark}"
    local tag="${3:-latest}"
    
    print_info "Building ${image_name}:${tag} using ${runtime}..."
    
    # Build the image with the benchmark name as build arg
    if $runtime build --no-cache \
        --build-arg BENCHMARK_NAME="${benchmark}" \
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
    
    print_info "Exgentic Benchmark Image Builder"
    print_info "================================="
    
    # Detect container runtime
    RUNTIME=$(detect_runtime)
    print_info "Detected container runtime: ${RUNTIME}"
    
    # Parse command line arguments
    BENCHMARK="${1}"
    TAG="${2:-latest}"
    
    # Validate benchmark name is provided
    if [ -z "$BENCHMARK" ]; then
        print_error "Benchmark name is required!"
        echo ""
        echo "Usage: $0 BENCHMARK [TAG]"
        echo ""
        echo "Available benchmarks: tau2, gsm8k"
        echo "Example: $0 tau2 v1.0.0"
        exit 1
    fi
    
    BENCHMARKS=("$BENCHMARK")
    
    print_info "Building benchmark: ${BENCHMARK}"
    print_info "Image tag: ${TAG}"
    echo ""
    
    # Build each benchmark
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    
    for benchmark in "${BENCHMARKS[@]}"; do
        if build_benchmark "$benchmark" "$RUNTIME" "$TAG"; then
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
        for benchmark in "${BENCHMARKS[@]}"; do
            echo "  - exgentic-mcp-${benchmark}:${TAG}"
        done
    fi
}

# Show usage if --help is passed
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    cat << EOF
Usage: $0 BENCHMARK [TAG]

Build exgentic benchmark Docker/Podman image.

Arguments:
  BENCHMARK   Benchmark name (required: tau2 or gsm8k)
  TAG         Image tag (optional, default: latest)

Examples:
  $0 tau2              # Build tau2 with 'latest' tag
  $0 gsm8k v1.0.0      # Build gsm8k with 'v1.0.0' tag
  $0 tau2 dev          # Build tau2 with 'dev' tag

Available benchmarks:
  - tau2
  - gsm8k

The script automatically detects whether to use docker or podman.
EOF
    exit 0
fi

main "$@"

# Made with Bob
