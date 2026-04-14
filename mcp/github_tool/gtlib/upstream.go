package lib

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/dgrijalva/jwt-go"
	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

// downstreamSessionID is for session IDs the gateway uses with its own clients
type downstreamSessionID string

// upstreamSessionID is for session IDs the gateway uses with upstream MCP servers
type upstreamSessionID string

// MCPUpstream manages an MCP server and its sessions
type MCPUpstream interface {
	// Call a tool by gatewaying to upstream MCP server.  Note that the upstream connection may be created lazily.
	CallTool(
		ctx context.Context,
		downstreamSession downstreamSessionID,
		request mcp.CallToolRequest,
	) (*mcp.CallToolResult, error)

	// Cleanup any upstream connections being held open on behalf of downstreamSessionID
	Close(ctx context.Context, downstreamSession downstreamSessionID) error

	// MCPServer gets an MCP server
	MCPServer() *server.MCPServer

	ToolsClient() *client.Client

	Tools() []mcp.Tool

	ListenAndServe() error

	// CreateSession creates a new MCP session for the given authority/host
	CreateSession(ctx context.Context, authority string) (string, error)

	// Shutdown closes any resources associated with this Broker
	Shutdown(ctx context.Context) error
}

// upstreamSessionState tracks what we manage about a connection an upstream MCP server
type upstreamSessionState struct {
	initialized bool
	client      *client.Client
	sessionID   upstreamSessionID
	lastContact time.Time
}

// upstreamMCP identifies what we know about an upstream MCP server
type upstreamMCP struct {
	mpcClient *client.Client // The MCP client we hold open to listen for tool notifications
	// initializeResult *mcp.InitializeResult // The init result when we probed at discovery time
	toolsResult *mcp.ListToolsResult // The tools when we probed at discovery time (or updated on toolsChanged notification)
}

// mcpBrokerImpl implements MCPBroker
type mcpAuthImpl struct {
	// URL of upstream
	url string

	// serverSessions tracks the sessions we maintain with the upstream MCP server
	serverSessions map[downstreamSessionID]*upstreamSessionState

	// mcpServer tracks the known server
	mcpServer upstreamMCP

	// listeningMCPServer returns an actual listening MCP server that federates registered MCP servers
	listeningMCPServer *server.MCPServer

	// The HTTP server used by listeningMCPServer
	httpServer *http.Server

	logger *slog.Logger
}

// Close implements MCPUpstream.
var _ MCPUpstream = &mcpAuthImpl{}

func MakeUpstream(mcpServerURL, initAuthHeader string) (*client.Client, context.CancelFunc, error) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))

	options := make([]transport.StreamableHTTPCOption, 0)
	options = append(options, transport.WithContinuousListening())
	options = append(options, transport.WithHTTPHeaders(map[string]string{
		"Authorization": initAuthHeader,
	}))
	httpClient, err := client.NewStreamableHttpClient(mcpServerURL, options...)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create client: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*24*time.Hour)
	err = httpClient.Start(ctx)
	if err != nil {
		defer cancel()
		return nil, nil, fmt.Errorf("failed to start streamable client %w", err)
	}

	capabilities := mcp.ClientCapabilities{
		Roots: &struct {
			ListChanged bool `json:"listChanged,omitempty"`
		}{
			ListChanged: true,
		},
	}
	clientName := "mcp-mitm"

	initResult, err := httpClient.Initialize(ctx, mcp.InitializeRequest{
		Params: mcp.InitializeParams{
			ProtocolVersion: mcp.LATEST_PROTOCOL_VERSION,
			Capabilities:    capabilities,
			ClientInfo: mcp.Implementation{
				Name:    clientName,
				Version: "0.0.1",
			},
		},
	})
	if err != nil {
		_ = httpClient.Close()
		defer cancel()
		return nil, nil, fmt.Errorf("initialization failed %w", err)
	}
	fmt.Printf("initResult=%v\n", initResult)

	clientTransport := httpClient.GetTransport()
	fmt.Printf("transport is %v, a %T\n", clientTransport, clientTransport)
	streamableHttpTransport, ok := clientTransport.(*transport.StreamableHTTP)
	if ok {
		streamableHttpTransport.SetNotificationHandler(func(notification mcp.JSONRPCNotification) {
			logger.Error("Notification Handler got", "notification", notification)
		})
	}

	err = httpClient.SetLevel(ctx, mcp.SetLevelRequest{
		Params: mcp.SetLevelParams{
			Level: mcp.LoggingLevelDebug,
		},
	})
	if err != nil {
		// (The server2 test server returns err="logging not supported")
		logger.Warn("failed to SetLevel", "err", err)
	}

	httpClient.OnConnectionLost(func(err error) {
		logger.Error("OnConnectionLost", "err", err)
	})

	httpClient.OnNotification(func(notification mcp.JSONRPCNotification) {
		logger.Error("OnNotification", "notification", notification)
	})

	return httpClient, cancel, nil
}

func ToolsToServerTools(mcpUpstream MCPUpstream, mcpURL string, newTools []mcp.Tool) []server.ServerTool {
	m, ok := mcpUpstream.(*mcpAuthImpl)
	if !ok {
		panic("mcpUpstream unknown impl")
	}

	tools := make([]server.ServerTool, 0)
	for _, newTool := range newTools {
		slog.Info("Federating tool", "mcpURL", mcpURL, "federated name", newTool.Name)
		tools = append(tools, toolToServerTool(m, newTool))
	}

	return tools
}

func toolToServerTool(m *mcpAuthImpl, newTool mcp.Tool) server.ServerTool {
	return server.ServerTool{
		Tool: newTool,
		Handler: func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			fmt.Printf("In Handler, request.Header is %#v, a %T\n", request.Header, request.Header)
			for k, v := range request.Header {
				fmt.Printf("Found header %s: %v\n", k, v)
			}
			result, err := m.CallTool(ctx,
				downstreamSessionID(request.Header.Get("Mcp-Session-Id")),
				request,
			)
			return result, err
		},
	}
}

func (m *mcpAuthImpl) CallTool(
	ctx context.Context,
	downstreamSession downstreamSessionID,
	request mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {

	upstreamSession, ok := m.serverSessions[downstreamSession]
	if !ok {

		incomingAuthHeader := request.Header.Get("Authorization")
		// auth options
		var options []transport.StreamableHTTPCOption
		serverAuthHeaderValue := m.getAuthorizationHeaderFromBearer(incomingAuthHeader)
		if serverAuthHeaderValue != "" {
			// slog.Info("Creating upstream session with authentication", "url", m.url)
			options = append(options, transport.WithHTTPHeaders(map[string]string{
				"Authorization": serverAuthHeaderValue,
			}))
		}

		var err error
		upstreamSession, err = m.createUpstreamSession(ctx, incomingAuthHeader, options...)
		if err != nil {
			return nil, fmt.Errorf("could not open upstream: %w", err)
		}
		m.serverSessions[downstreamSession] = upstreamSession
	}

	if args, ok := request.Params.Arguments.(map[string]interface{}); ok {
		stripNilArguments(args)
	}

	res, err := upstreamSession.client.CallTool(ctx, request)
	if err != nil {
		upstreamSession.lastContact = time.Now()
	}

	return res, err
}

// stripNilArguments removes nil-valued entries from tool call arguments.
// OpenAI's function calling API sends explicit JSON null for optional
// parameters, but the upstream GitHub MCP server rejects nil where it
// expects a typed value.
func stripNilArguments(args map[string]interface{}) {
	for k, v := range args {
		if v == nil {
			delete(args, k)
		}
	}
}

func (m *mcpAuthImpl) createUpstreamSession(ctx context.Context, authorization string, options ...transport.StreamableHTTPCOption) (*upstreamSessionState, error) {
	retval := &upstreamSessionState{}

	var err error
	retval.client, _, err = m.createMCPClient(ctx, authorization, options...)
	if err != nil {
		return nil, err
	}

	retval.initialized = true
	retval.sessionID = upstreamSessionID(retval.client.GetSessionId())
	retval.lastContact = time.Now()

	return retval, nil
}
func (m *mcpAuthImpl) Close(ctx context.Context, downstreamSession downstreamSessionID) error {
	serverSession, ok := m.serverSessions[downstreamSession]
	if !ok {
		return fmt.Errorf("unknown session %q", downstreamSession)
	}

	_ = serverSession.client.Close()
	delete(m.serverSessions, downstreamSession)
	return nil
}

// CreateSession implements MCPUpstream.
func (m *mcpAuthImpl) CreateSession(ctx context.Context, authority string) (string, error) {
	panic("unimplemented")
}

// MCPServer implements MCPUpstream.
func (m *mcpAuthImpl) MCPServer() *server.MCPServer {
	return m.listeningMCPServer
}

// Shutdown implements MCPUpstream.
func (m *mcpAuthImpl) Shutdown(ctx context.Context) error {
	// TODO close all upstreams and server
	return nil
}

// createMCPClient creates and initializes an MCP client with the appropriate configuration
func (m *mcpAuthImpl) createMCPClient(ctx context.Context, authorization string, options ...transport.StreamableHTTPCOption) (*client.Client, *mcp.InitializeResult, error) {
	// Use the registered upstream with its CredentialEnvVar
	authHeader := m.getAuthorizationHeaderFromBearer(authorization)
	if authHeader != "" {
		// slog.Info("Creating upstream session with authentication", "authHeader", authHeader)
		options = append(options, transport.WithHTTPHeaders(map[string]string{
			"Authorization": authHeader,
		}))
	}

	httpClient, err := client.NewStreamableHttpClient(m.url, options...)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create client: %w", err)
	}

	capabilities := mcp.ClientCapabilities{
		Roots: &struct {
			ListChanged bool `json:"listChanged,omitempty"`
		}{
			ListChanged: true,
		},
	}
	clientName := "kagenti-mitm-mcp"

	initResp, err := httpClient.Initialize(ctx, mcp.InitializeRequest{
		Params: mcp.InitializeParams{
			ProtocolVersion: mcp.LATEST_PROTOCOL_VERSION,
			Capabilities:    capabilities,
			ClientInfo: mcp.Implementation{
				Name:    clientName,
				Version: "0.0.1",
			},
		},
	})
	if err != nil {
		_ = httpClient.Close()
		return nil, nil, fmt.Errorf("initialization failed: %w", err)
	}

	return httpClient, initResp, nil
}

// Tools implements MCPUpstream.
func (m *mcpAuthImpl) Tools() []mcp.Tool {
	return m.mcpServer.toolsResult.Tools
}

// ToolsClient implements MCPUpstream.
func (m *mcpAuthImpl) ToolsClient() *client.Client {
	return m.mcpServer.mpcClient
}

// ListenAndServe implements MCPUpstream.
func (m *mcpAuthImpl) ListenAndServe() error {
	return m.httpServer.ListenAndServe()
}

// In this proof-of-concept we do this explicitly in our logic, instead of using middleware, to make the control flow obvious.
func (m *mcpAuthImpl) getAuthorizationHeaderFromBearer(auth string) string {
	// fmt.Printf("incoming auth header is %q\n", auth)
	if !strings.HasPrefix(auth, "Bearer ") {
		fmt.Printf("Oops, the man-in-the-middle didn't get an Auth header starting with 'Bearer '!")
		return ""
	}

	token := strings.TrimPrefix(auth, "Bearer ")

	// For explanation see https://stackoverflow.com/questions/55698770/decode-jwt-without-validation-and-find-scope
	jwtToken, _, err := new(jwt.Parser).ParseUnverified(token, jwt.MapClaims{})
	if err != nil {
		fmt.Printf("Bearer token couldn't be parsed: %v\n", err)
		return ""
	}

	// TODO check that jwtToken hasn't expired, is valid, etc.

	claims, ok := jwtToken.Claims.(jwt.MapClaims)
	if !ok {
		fmt.Printf("Parsed token not MapClaims: %v\n", err)
		return ""
	}

	// fmt.Printf("Got claims %#v\n", claims)
	decodedToken := claims

	/*
		x := map[string]interface {}{
			"acr":"1",
			"allowed-origins":[]interface {}{"http://kagenti-ui.localtest.me:8080"},
			"aud":[]interface {}{"demo-realm", "master-realm", "account"},
			"azp":"kagenti",
			"email_verified":false,
			"exp":1.760644947e+09,
			"iat":1.760644887e+09,
			"iss":"http://keycloak.localtest.me:8080/realms/master",
			"jti":"onrtro:cf3ef4dc-22b8-7d78-646d-784278b3b448",
			"preferred_username":"admin",
			"realm_access":map[string]interface {}{
				"roles":[]interface {}{"create-realm", "default-roles-master", "offline_access", "admin", "uma_authorization"}
			},
			"resource_access":map[string]interface {}{
				"account":map[string]interface {}{
					"roles":[]interface {}{"manage-account", "manage-account-links", "view-profile"}
				},
				"demo-realm":map[string]interface {}{
					"roles":[]interface {}{"view-realm", "view-identity-providers", "manage-identity-providers", "impersonation", "create-client", "manage-users", "query-realms", "view-authorization", "query-clients", "query-users", "manage-events", "manage-realm", "view-events", "view-users", "view-clients", "manage-authorization", "manage-clients", "query-groups"}
				},
				"master-realm":map[string]interface {}{
					"roles":[]interface {}{"view-realm", "view-identity-providers", "manage-identity-providers", "impersonation", "create-client", "manage-users", "query-realms", "view-authorization", "query-clients", "query-users", "manage-events", "manage-realm", "view-events", "view-users", "view-clients", "manage-authorization", "manage-clients", "query-groups"}
				}},
				"scope":"email profile",
				"sid":"8b4ebb18-c6ef-4e44-b70e-71d67a6ef8ed", "sub":"8e805ce6-5621-4d89-a7af-fcce52f5ae43", "typ":"Bearer"}
		_ = x
	*/

	// for key := range maps.Keys(decodedToken) {
	// 	fmt.Printf("decodedToken has key %v\n", key)
	// }

	scopesObj, ok := decodedToken["scope"]
	if !ok {
		m.logger.Info("decoded token had no scopes", "decodedToken", decodedToken)
		return ""
	}
	scopesStr, ok := scopesObj.(string)
	if !ok {
		m.logger.Info("decoded token scopes wasn't a string")
		return ""
	}
	fmt.Printf("This OIDC user has scopes %q\n", scopesStr)
	scopes := strings.Split(scopesStr, " ")
	requiredScope := os.Getenv("REQUIRED_SCOPE")
	scopeMatches := slices.Contains(scopes, requiredScope)

	// This makes sense to me, but isn't what Maia requested:
	// if !scopeMatches {
	// 	return ""
	// }
	// return os.Getenv(fmt.Sprintf("GITHUB_TOKEN_for_%s", decodedToken["preferred_username"]))

	if scopeMatches {
		fmt.Printf("The REQUIRED_SCOPE %q in scopes %v\n", requiredScope, scopes)
		return os.Getenv("UPSTREAM_HEADER_TO_USE_IF_IN_AUDIENCE")
	} else {
		fmt.Printf("The REQUIRED_SCOPE %q NOT IN scopes %v\n", requiredScope, scopes)
		return os.Getenv("UPSTREAM_HEADER_TO_USE_IF_NOT_IN_AUDIENCE")
	}

	/*
		This is the start of the secure implementation
		// Use /.well-known/openid-configuration/jwks ?  Ask Keycloak to decode?  Prefetch/cache public keys?

		// Parse the token with CustomClaims
		claims := jwt.MapClaims{}
		parsedToken, err := jwt.ParseWithClaims(token, claims, func(token *jwt.Token) (interface{}, error) {

			// Validate the signing method and return the secret key
			if hmacToken, ok := token.Method.(*jwt.SigningMethodHMAC); ok {
				fmt.Printf("Got HMAC token %#v\n", hmacToken)
				secretKey := "a-string-secret-at-least-256-bits-long" // TODO
				return secretKey, nil
			}

			// FYI Keycloak bearer tokens are signed with jwt.SigningMethodRSA, not SigningMethodHMAC
			if rsaToken, ok := token.Method.(*jwt.SigningMethodRSA); ok {
				fmt.Printf("Got RSA token %#v\n", rsaToken)

				// A fast way to validate is to follow the steps of
				// https://stackoverflow.com/questions/77838958/go-validate-access-token-keycloak
				// and keep the public keys in memory.

				// Extract the key ID (kid) from the token header
				kid, ok := token.Header["kid"].(string)
				if !ok {
					return nil, fmt.Errorf("kid not found in token header")
				}
				_ = kid
				return &rsa.PublicKey{}, nil
			}

			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		})
		if err != nil {
			fmt.Printf("Could not parse JWT, failed with %v\n", err)
		} else {
			fmt.Printf("Parsed into parsedToken %#v, a %T\n", parsedToken, parsedToken)
			fmt.Printf("Claims is %#v\n", claims)
		}

		return "" // fail
	*/
}
