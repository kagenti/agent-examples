# Shopping Agent Architecture

## System Overview

The Shopping Agent is a sophisticated MCP (Model Context Protocol) server that uses LangChain, LangGraph, OpenAI, and SerpAPI to provide intelligent product recommendations.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client Application                           │
│                    (Any MCP-compatible client)                       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTP/MCP Protocol
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Shopping Agent MCP Server                       │
│                         (FastMCP Framework)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌───────────────────┐         ┌──────────────────┐                │
│  │ recommend_products│         │ search_products  │                │
│  │      @mcp.tool    │         │    @mcp.tool     │                │
│  └─────────┬─────────┘         └─────────┬────────┘                │
│            │                              │                          │
│            └──────────────┬───────────────┘                          │
│                           ▼                                          │
│            ┌──────────────────────────┐                             │
│            │   LangGraph Agent Core   │                             │
│            │   (Workflow Orchestrator)│                             │
│            └──────────┬───────────────┘                             │
│                       │                                              │
└───────────────────────┼──────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
   ┌─────────┐   ┌──────────┐   ┌─────────────┐
   │  Parse  │   │  Search  │   │  Generate   │
   │  Query  │   │ Products │   │Recommenda-  │
   │  Node   │   │   Node   │   │  tions Node │
   └────┬────┘   └─────┬────┘   └──────┬──────┘
        │              │               │
        │              │               │
        ▼              ▼               ▼
  ┌─────────┐    ┌─────────┐    ┌─────────┐
  │ OpenAI  │    │ SerpAPI │    │ OpenAI  │
  │   API   │    │   API   │    │   API   │
  └─────────┘    └─────────┘    └─────────┘
```

## Component Details

### 1. FastMCP Server Layer

**Purpose**: Exposes tools via Model Context Protocol

**Components**:
- `FastMCP("Shopping Agent")`: MCP server instance
- Tool decorators with proper annotations
- HTTP transport support (MCP Inspector compatible)
- Environment-based configuration

**Key Features**:
- RESTful API endpoints for tools
- Tool metadata and documentation
- Error handling and validation
- Logging and monitoring

### 2. LangGraph Agent Core

**Purpose**: Orchestrates multi-step agent workflow with state management

**State Definition**:
```python
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]      # Conversation history
    search_results: List[Dict[str, Any]]         # Raw search data
    recommendations: List[Dict[str, Any]]         # Final recommendations
```

**Workflow Graph**:
```
START
  ↓
[parse_query]
  ↓
[search_products]
  ↓
[generate_recommendations]
  ↓
END
```

**Node Functions**:
1. **parse_query_node**: Extracts structured data from natural language
2. **search_products_node**: Performs product search via SerpAPI
3. **generate_recommendations_node**: Creates curated recommendations

### 3. Node Implementations

#### Parse Query Node

```
Input:  "I want to buy a scarf for 40 dollars"
        ↓
    [OpenAI GPT-4o-mini]
        ↓
Output: {product: "scarf", budget: "40"}
```

**Process**:
1. Receives user query in natural language
2. Uses OpenAI to extract structured information
3. Identifies product type and budget constraints
4. Updates state with parsed data

#### Search Products Node

```
Input:  {product: "scarf", budget: "40"}
        ↓
    [SerpAPI Search]
        ↓
Output: Raw search results from retailers
```

**Process**:
1. Constructs optimized search query
2. Queries SerpAPI for product listings
3. Aggregates results from multiple sources
4. Stores raw results in state

#### Generate Recommendations Node

```
Input:  Raw search results
        ↓
    [OpenAI GPT-4o-mini]
        ↓
Output: Structured recommendations (max 5)
```

**Process**:
1. Analyzes search results using OpenAI
2. Extracts product details (name, price, description)
3. Generates reasoning for each recommendation
4. Formats as structured JSON response
5. Limits to requested number of recommendations

## Data Flow

### Request Flow

```
1. Client sends query
   ↓
2. MCP server receives request
   ↓
3. Tool validates parameters
   ↓
4. LangGraph agent initialized
   ↓
5. State created with initial message
   ↓
6. Workflow executes sequentially:
   a. Parse query (OpenAI)
   b. Search products (SerpAPI)
   c. Generate recommendations (OpenAI)
   ↓
7. Final state extracted
   ↓
8. Recommendations formatted as JSON
   ↓
9. Response sent to client
```

### State Transitions

```
Initial State:
{
  messages: [HumanMessage("I want a scarf for $40")],
  search_results: [],
  recommendations: []
}
    ↓ [parse_query]
{
  messages: [..., AIMessage("Searching for scarf within budget $40...")],
  search_results: [],
  recommendations: []
}
    ↓ [search_products]
{
  messages: [...],
  search_results: [{raw_results: "...search data..."}],
  recommendations: []
}
    ↓ [generate_recommendations]
{
  messages: [..., AIMessage("Here are my top recommendations...")],
  search_results: [...],
  recommendations: [
    {name: "Product 1", price: "$38.99", ...},
    {name: "Product 2", price: "$39.99", ...},
    ...
  ]
}
```

## Integration Points

### External APIs

#### OpenAI API
- **Usage**: Natural language understanding and generation
- **Model**: gpt-4o-mini
- **Calls per request**: 2 (parse + generate)
- **Authentication**: API key via environment variable

#### SerpAPI
- **Usage**: Real-time product search
- **Calls per request**: 1
- **Authentication**: API key via environment variable
- **Results**: Aggregated from multiple retailers

### LangChain Components

```
langchain_openai.ChatOpenAI
    ├── Model configuration
    ├── Temperature settings
    └── API key management

langchain_community.utilities.SerpAPIWrapper
    ├── Search query optimization
    ├── Result parsing
    └── API key management

langchain.schema
    ├── HumanMessage
    ├── AIMessage
    └── SystemMessage
```

### LangGraph Components

```
langgraph.graph.StateGraph
    ├── Node definitions
    ├── Edge connections
    ├── Entry/exit points
    └── State management

langgraph.graph.message.add_messages
    └── Message history reducer
```

## Error Handling

### Error Flow

```
Try:
    Execute workflow
    ↓
    [Node execution]
    ↓
    Check for errors
Except APIError:
    Log error
    Return structured error response
Except ValidationError:
    Log error
    Return validation error
Except Exception:
    Log with traceback
    Return generic error
Finally:
    Clean up resources
```

### Error Types

1. **API Key Errors**: Missing or invalid API keys
2. **API Quota Errors**: Rate limits exceeded
3. **Network Errors**: Connection failures
4. **Parsing Errors**: Invalid JSON responses
5. **Validation Errors**: Invalid parameters

## Performance Characteristics

### Latency Breakdown

```
Total Request Time: ~5-10 seconds
    ├── Parse Query: ~1-2 seconds (OpenAI)
    ├── Search Products: ~2-3 seconds (SerpAPI)
    └── Generate Recommendations: ~2-5 seconds (OpenAI)
```

### Optimization Strategies

1. **Parallel API Calls**: Future enhancement to call OpenAI and SerpAPI in parallel where possible
2. **Caching**: Cache common searches
3. **Result Limiting**: Limit search results to reduce processing time
4. **Model Selection**: Use gpt-4o-mini for cost-effective performance

## Scalability

### Horizontal Scaling

```
Load Balancer
    ↓
┌───────────┬───────────┬───────────┐
│ Instance 1│ Instance 2│ Instance 3│
└───────────┴───────────┴───────────┘
```

### Considerations

- Stateless design (no session storage)
- Independent request processing
- External API rate limits (OpenAI, SerpAPI)
- Docker containerization for easy deployment

## Security

### API Key Management

```
Environment Variables
    ├── OPENAI_API_KEY (required)
    ├── SERPAPI_API_KEY (required)
    └── Never logged or exposed
```

### Input Validation

- Query length limits
- maxResults bounds checking
- Parameter type validation
- Error message sanitization

## Monitoring and Logging

### Log Levels

```
DEBUG:   Detailed execution flow
INFO:    Important state changes
WARNING: Recoverable issues
ERROR:   Failures and exceptions
```

### Key Metrics

- Request count
- Response times
- API call counts
- Error rates
- Recommendation quality

## Deployment Architecture

### Docker Deployment

```
Docker Container
    ├── Python 3.11
    ├── uv package manager
    ├── Application code
    ├── Dependencies
    └── Environment configuration

Exposed:
    └── Port 8000 (HTTP)

Environment:
    ├── OPENAI_API_KEY
    ├── SERPAPI_API_KEY
    ├── HOST (0.0.0.0)
    ├── PORT (8000)
    └── LOG_LEVEL (INFO)
```

### Production Setup

```
┌─────────────────┐
│  Load Balancer  │
└────────┬────────┘
         │
    ┌────┴────┬────────┐
    ▼         ▼        ▼
┌────────┐ ┌────────┐ ┌────────┐
│Container│ │Container│ │Container│
│   #1   │ │   #2   │ │   #3   │
└────────┘ └────────┘ └────────┘
```

## Future Enhancements

### Planned Features

1. **Caching Layer**: Redis for common searches
2. **Advanced Filtering**: Price ranges, categories, ratings
3. **User Preferences**: Remember user preferences
4. **Multiple Providers**: Add more search providers
5. **Review Integration**: Include product reviews
6. **Price Tracking**: Track price changes over time
7. **Comparison Mode**: Side-by-side product comparison
8. **Image Analysis**: Use vision models for product images

### Architectural Improvements

1. **Async/Await**: Non-blocking API calls
2. **Streaming Responses**: Stream recommendations as they're found
3. **Graph Optimization**: Parallel node execution where possible
4. **Advanced State Management**: More sophisticated state tracking
5. **Tool Chaining**: Compose multiple tools together

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Protocol** | FastMCP | MCP server framework |
| **Orchestration** | LangGraph | Agent workflow management |
| **LLM Framework** | LangChain | LLM integration and utilities |
| **NLU/NLG** | OpenAI GPT-4o-mini | Language understanding/generation |
| **Search** | SerpAPI | Product search across retailers |
| **Runtime** | Python 3.11 | Application runtime |
| **Package Manager** | uv | Fast Python package management |
| **Containerization** | Docker | Deployment and isolation |

## Conclusion

The Shopping Agent represents a production-ready implementation of:
- ✅ Modern MCP server patterns
- ✅ Advanced LangGraph agent workflows
- ✅ Multi-API integration
- ✅ Robust error handling
- ✅ Scalable architecture
- ✅ Comprehensive documentation

Ready for deployment and real-world use! 🚀

