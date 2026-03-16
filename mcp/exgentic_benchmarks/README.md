# Exgentic Benchmarks MCP Server

A Docker-based MCP (Model Context Protocol) server that runs the Exgentic benchmark system. This server clones the Exgentic repository, installs a specific benchmark at build time, and exposes it via the MCP protocol.

## Overview

This MCP server provides access to Exgentic benchmarks through a standardized interface. Each Docker image is built with a specific benchmark pre-installed, making it easy to deploy and run different benchmarks in isolated environments.

## Features

- **Benchmark-Specific Images**: Each Docker image contains a single benchmark
- **Build-Time Setup**: Benchmarks are installed during image build for faster startup
- **Flexible Configuration**: HOST and PORT configurable at runtime via environment variables
- **Consistent Interface**: Follows MCP protocol standards
- **Security**: Runs as non-root user (UID 1001)
- **Production Ready**: Includes proper error handling and logging

## Prerequisites

- Docker installed on your system
- Internet connection (for cloning repository and downloading benchmark data)
- Sufficient disk space (benchmark data can be large)

## Quick Start

### Build the Docker Image

Build an image with the `tau2` benchmark:

```bash
cd mcp/exgentic_benchmarks

docker build \
  --build-arg BENCHMARK_NAME=tau2 \
  -t exgentic-mcp-tau2:latest \
  .
```

### Run the Server

```bash
docker run -p 8000:8000 exgentic-mcp-tau2:latest
```

The server will start on `http://0.0.0.0:8000`

### Test the Server

```bash
# Check if server is running
curl http://localhost:8000/health

# List available tools
curl http://localhost:8000/tools
```

## Building for Different Benchmarks

You can build images for different benchmarks by changing the `BENCHMARK_NAME` argument:

### WebArena Benchmark
```bash
docker build \
  --build-arg BENCHMARK_NAME=webarena \
  -t exgentic-mcp-webarena:latest \
  .
```

### MiniWoB Benchmark
```bash
docker build \
  --build-arg BENCHMARK_NAME=miniwob \
  -t exgentic-mcp-miniwob:latest \
  .
```

### Tau2 Benchmark (Default Example)
```bash
docker build \
  --build-arg BENCHMARK_NAME=tau2 \
  -t exgentic-mcp-tau2:latest \
  .
```

## Configuration

### Build Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `BENCHMARK_NAME` | Yes | - | The benchmark to install (e.g., tau2, webarena, miniwob) |
| `RELEASE_VERSION` | No | main | Version tag for tracking |

### Runtime Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | 0.0.0.0 | Server host address |
| `PORT` | 8000 | Server port |
| `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `BENCHMARK_NAME` | (from build) | Benchmark name (set during build) |

### Custom Configuration Examples

**Run on a different port:**
```bash
docker run -p 9000:9000 \
  -e PORT=9000 \
  exgentic-mcp-tau2:latest
```

**Enable debug logging:**
```bash
docker run -p 8000:8000 \
  -e LOG_LEVEL=DEBUG \
  exgentic-mcp-tau2:latest
```

**Bind to specific host:**
```bash
docker run -p 8000:8000 \
  -e HOST=127.0.0.1 \
  exgentic-mcp-tau2:latest
```

## Architecture

### Build Process

```
1. Clone Exgentic repository (HTTPS)
   ↓
2. Checkout feature/mcp-command branch
   ↓
3. Install Exgentic and dependencies
   ↓
4. Run: exgentic setup --benchmark $BENCHMARK_NAME
   ↓
5. Configure entrypoint and permissions
   ↓
6. Create image: exgentic-mcp-{benchmark}:latest
```

### Runtime Process

```
1. Container starts with entrypoint.sh
   ↓
2. Read environment variables (HOST, PORT)
   ↓
3. Execute: exgentic mcp --benchmark $BENCHMARK_NAME --host $HOST --port $PORT
   ↓
4. MCP server listens on configured HOST:PORT
```

## Image Naming Convention

Images follow the pattern: `exgentic-mcp-{benchmark}:latest`

Examples:
- `exgentic-mcp-tau2:latest`
- `exgentic-mcp-webarena:latest`
- `exgentic-mcp-miniwob:latest`

## Advanced Usage

### Docker Compose

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  exgentic-tau2:
    build:
      context: .
      args:
        BENCHMARK_NAME: tau2
    ports:
      - "8000:8000"
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - LOG_LEVEL=INFO
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

### Multi-Benchmark Deployment

Run multiple benchmarks simultaneously on different ports:

```bash
# Tau2 on port 8000
docker run -d -p 8000:8000 --name exgentic-tau2 exgentic-mcp-tau2:latest

# WebArena on port 8001
docker run -d -p 8001:8001 -e PORT=8001 --name exgentic-webarena exgentic-mcp-webarena:latest

# MiniWoB on port 8002
docker run -d -p 8002:8002 -e PORT=8002 --name exgentic-miniwob exgentic-mcp-miniwob:latest
```

### Volume Mounting (Optional)

If you need to persist data or share files:

```bash
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  exgentic-mcp-tau2:latest
```

## Troubleshooting

### Build Fails

**Problem**: `BENCHMARK_NAME build argument is required`
```bash
# Solution: Always provide BENCHMARK_NAME
docker build --build-arg BENCHMARK_NAME=tau2 -t exgentic-mcp-tau2 .
```

**Problem**: Git clone fails
```bash
# Solution: Check internet connection and GitHub access
# The repository is public and uses HTTPS, so no authentication needed
```

**Problem**: Benchmark setup fails
```bash
# Solution: Check if the benchmark name is valid
# Verify the benchmark exists in the Exgentic repository
# Check build logs for specific error messages
```

### Runtime Issues

**Problem**: Port already in use
```bash
# Solution: Use a different port
docker run -p 9000:9000 -e PORT=9000 exgentic-mcp-tau2:latest
```

**Problem**: Container exits immediately
```bash
# Solution: Check logs
docker logs <container_id>

# Run in foreground to see errors
docker run -it exgentic-mcp-tau2:latest
```

**Problem**: Cannot connect to server
```bash
# Solution: Verify port mapping and firewall
docker ps  # Check if container is running
curl http://localhost:8000/health  # Test connection
```

### Debugging

**View container logs:**
```bash
docker logs -f <container_name>
```

**Access container shell:**
```bash
docker exec -it <container_name> /bin/bash
```

**Check running processes:**
```bash
docker exec <container_name> ps aux
```

## Development

### Local Development

For development, you can mount the local Exgentic repository:

```bash
docker run -p 8000:8000 \
  -v /path/to/local/exgentic:/app/exgentic \
  exgentic-mcp-tau2:latest
```

### Rebuilding

After making changes, rebuild the image:

```bash
docker build --no-cache \
  --build-arg BENCHMARK_NAME=tau2 \
  -t exgentic-mcp-tau2:latest \
  .
```

## Repository Information

- **Source**: https://github.com/Exgentic/exgentic.git
- **Branch**: feature/mcp-command
- **Protocol**: HTTPS (public repository)

## MCP Protocol

This server implements the Model Context Protocol (MCP), which allows AI assistants to discover and use benchmark tools programmatically.

### Key Features

- **Tool Discovery**: Benchmarks expose their capabilities via MCP
- **Type Safety**: Strong typing for parameters and returns
- **Documentation**: Built-in documentation for each tool
- **Error Handling**: Standardized error responses
- **Transport**: HTTP transport with streamable support

## Security Considerations

- Runs as non-root user (UID 1001)
- No SSH keys or secrets in image
- Public repository access only
- Minimal attack surface
- Production-ready configuration

## Performance Notes

- **Build Time**: 5-15 minutes depending on benchmark size
- **Image Size**: Varies by benchmark (typically 1-5 GB)
- **Startup Time**: Fast (benchmark already installed)
- **Memory**: Depends on benchmark requirements

## Contributing

When adding support for new benchmarks:

1. Verify the benchmark exists in the Exgentic repository
2. Test the build process
3. Document any special requirements
4. Update this README with examples

## License

See the repository's LICENSE file for details.

## Support

For issues related to:
- **This MCP server**: Open an issue in the agent-examples repository
- **Exgentic benchmarks**: Refer to the Exgentic repository
- **MCP protocol**: See the Model Context Protocol documentation

## Related Resources

- [Exgentic Repository](https://github.com/Exgentic/exgentic)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Docker Documentation](https://docs.docker.com/)
- [Other MCP Tools](../README.md)