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
    local tag=$3
    local use_cache=$4
    
    print_info "Building ${image_name}:${tag} using ${runtime}..."
    if [ "$use_cache" = "false" ]; then
        print_info "Building without cache (default)"
    else
        print_info "Building with cache enabled"
    fi
    
    # Build the image with the benchmark name as build arg
    local build_cmd="$runtime build"
    if [ "$use_cache" = "false" ]; then
        build_cmd="$build_cmd --no-cache"
    fi
    
    if $build_cmd \
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
    BENCHMARK=""
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
Usage: $0 BENCHMARK [--tag TAG] [--use-cache]

Build exgentic benchmark Docker/Podman image.

Arguments:
  BENCHMARK      Benchmark name (required, positional: tau2 or gsm8k)
  --tag TAG      Image tag (optional, default: latest)
  --use-cache    Use Docker cache during build (optional, default: no cache for consistency)

Examples:
  $0 tau2                      # Build without cache (default)
  $0 gsm8k --tag v1.0.0        # Build v1.0.0 without cache
  $0 tau2 --use-cache          # Build with cache enabled
  $0 gsm8k --tag v1.0.0 --use-cache  # Build v1.0.0 with cache

Available benchmarks:
  - tau2
  - gsm8k

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
                if [ -z "$BENCHMARK" ]; then
                    BENCHMARK="$1"
                    shift
                else
                    print_error "Unexpected argument: $1"
                    echo "Use --help for usage information"
                    exit 1
                fi
                ;;
        esac
    done
    
    # Validate benchmark name is provided
    if [ -z "$BENCHMARK" ]; then
        print_error "Benchmark name is required!"
        echo ""
        echo "Usage: $0 BENCHMARK [--tag TAG] [--no-cache]"
        echo ""
        echo "Available benchmarks: tau2, gsm8k"
        echo "Example: $0 tau2 --tag v1.0.0"
        echo "Use --help for more information"
        exit 1
    fi
    
    BENCHMARKS=("$BENCHMARK")
    
    print_info "Building benchmark: ${BENCHMARK}"
    print_info "Image tag: ${TAG}"
    if [ "$USE_CACHE" = "true" ]; then
        print_info "Cache: enabled"
    else
        print_info "Cache: disabled (default)"
    fi
    echo ""
    
    # Build each benchmark
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    
    for benchmark in "${BENCHMARKS[@]}"; do
        if build_benchmark "$benchmark" "$RUNTIME" "$TAG" "$USE_CACHE"; then
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

main "$@"

# Made with Bob
