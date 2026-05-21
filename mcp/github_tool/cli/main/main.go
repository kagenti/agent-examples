package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"time"

	lib "github.com/kagenti/mcp-mitm/gtlib"
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))

	mcpServerURL := os.Getenv("UPSTREAM_MCP")
	if mcpServerURL == "" {
		mcpServerURL = "https://api.githubcopilot.com/mcp/"
	}
	logger.Info("Using MCP Server", "UPSTREAM_MCP", os.Getenv("UPSTREAM_MCP"), "mcpServerURL", mcpServerURL)

	initAuthHeader := os.Getenv("INIT_AUTH_HEADER")
	if initAuthHeader == "" {
		fmt.Fprintf(os.Stderr, "You must supply INIT_AUTH_HEADER=\"Bearer ${GITHUB_TOKEN}\"\n")
		os.Exit(1)
	}
	logger.Info("Using Authorization header", "initAuthHeader", initAuthHeader)

	mcpServer, err := lib.MakeMCPServer(mcpServerURL, initAuthHeader, listenerHost(), listenerPort())
	if err != nil {
		fmt.Fprintf(os.Stderr, "Can't server MCP: %v", err)
		os.Exit(4)
	}

	// We ignore if the MCP server reports tools changing later
	mcpServer.MCPServer().AddTools(lib.ToolsToServerTools(mcpServer, mcpServerURL, mcpServer.Tools())...)

	go func() {
		logger.Info("[http] starting MCP Broker (public)", "listening", listenerAddress())
		if err := mcpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("[http] Cannot start MCP server", "err", err)
		}
		fmt.Printf("ListenAndServe returned %v\n", err)
	}()

	ctx := context.Background()
	for {
		err = mcpServer.ToolsClient().Ping(ctx)
		if err != nil {
			logger.Error("failed to Ping", "err", err)
		} else {
			logger.Info("Ping!")
		}
		time.Sleep(5 * time.Second)
	}
}

func listenerHost() string {
	listenAddr := os.Getenv("LISTENER_HOST")
	if listenAddr != "" {
		return listenAddr
	}

	return "0.0.0.0"
}

func listenerPort() int {
	listenPort := os.Getenv("LISTENER_PORT")
	if listenPort != "" {
		port, err := strconv.Atoi(listenPort)
		if err != nil {
			panic(fmt.Sprintf("$LISTENER_PORT must be a port number, got %q", port))
		}
		return port
	}
	return 9090
}

func listenerAddress() string {
	return fmt.Sprintf("%s:%d", listenerHost(), listenerPort())
}
