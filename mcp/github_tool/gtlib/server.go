package lib

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

func MakeMCPServer(mcpServerURL, initAuthHeader, listenAddr string, listenPort int) (MCPUpstream, error) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))

	httpClient, cancel, err := MakeUpstream(mcpServerURL, initAuthHeader)
	if err != nil {
		logger.Error("Can't fetch tools from upstream", "mcpServerURL", mcpServerURL, "err", err)
		os.Exit(3)
	}
	defer cancel()

	ctx := context.Background()

	resTools, err := httpClient.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		defer cancel()
		logger.Error("failed to list tools", "err", err)
		os.Exit(4)
	}
	logger.Debug("MakeMCPServer listed resTools", "resTools", resTools)

	mcpServer, httpSrv, err := MakeDownstream(listenAddr, listenPort)
	if err != nil {
		return nil, err
	}

	return &mcpAuthImpl{
		url:            mcpServerURL,
		serverSessions: map[downstreamSessionID]*upstreamSessionState{},
		mcpServer: upstreamMCP{
			mpcClient:   httpClient,
			toolsResult: resTools,
		},
		listeningMCPServer: mcpServer,
		httpServer:         httpSrv,
		logger:             logger,
	}, nil
}

func MakeDownstream(listenAddr string, listenPort int) (*server.MCPServer, *http.Server, error) {
	hooks := &server.Hooks{}

	hooks.AddOnUnregisterSession(func(_ context.Context, session server.ClientSession) {
		slog.Info("Client disconnected", "sessionID", session.SessionID())
	})

	// Enhanced session registration to log gateway session assignment
	hooks.AddOnRegisterSession(func(_ context.Context, session server.ClientSession) {
		// Note that AddOnRegisterSession is for GET, not POST, for a session.
		// https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#listening-for-messages-from-the-server
		slog.Info("Gateway client connected with session", "gatewaySessionID", session.SessionID())
	})

	hooks.AddBeforeAny(func(_ context.Context, _ any, method mcp.MCPMethod, _ any) {
		slog.Info("Processing request", "method", method)
	})

	hooks.AddOnError(func(_ context.Context, _ any, method mcp.MCPMethod, _ any, err error) {
		slog.Info("MCP server error", "method", method, "error", err)
	})

	mux := http.NewServeMux()
	httpSrv := &http.Server{
		Addr:         fmt.Sprintf("%s:%d", listenAddr, listenPort),
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
	}
	slog.Info("MakeDownstream created http.Serve", "httpSrv.Addr", httpSrv.Addr)

	listeningMCPServer := server.NewMCPServer(
		"MitM MCP Broker",
		"0.0.1",
		server.WithHooks(hooks),
		server.WithToolCapabilities(true),
	)

	streamableHTTPServer := server.NewStreamableHTTPServer(
		listeningMCPServer,
		server.WithStreamableHTTPServer(httpSrv),
	)

	mux.Handle("/mcp", streamableHTTPServer)

	fmt.Printf("Will listen for MCP on %s:%d\n", listenAddr, listenPort)

	return listeningMCPServer, httpSrv, nil
}
