# Shopping Agent Architecture

## System Overview

The Shopping Agent is an MCP (Model Context Protocol) server that uses SerpAPI to provide product search capabilities. The server returns structured product data, and the calling agent (your AI) provides intelligent analysis and recommendations.

## Design Philosophy

**Separation of Concerns:**
- **MCP Server**: Provides data (product search results)
- **AI Agent**: Provides intelligence (analysis and recommendations)

This architecture follows the principle that MCP servers should be lightweight data providers, while AI agents handle complex reasoning and decision-making.

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
│            │    SerpAPI Integration   │                             │
│            │   (Product Search Only)  │                             │
│            └──────────┬───────────────┘                             │
│                       │                                              │
└───────────────────────┼──────────────────────────────────────────────┘
                        │
                        ▼
                   ┌─────────┐
                   │ SerpAPI │
                   │   API   │
                   └─────────┘

                        ↓
        Returns structured product data to agent
                        ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       AI Agent (Your Client)                         │
│                   Analyzes data & provides reasoning                 │
│                                                                       │
│  • Evaluates products against user requirements                      │
│  • Considers budget constraints                                      │
│  • Provides recommendation scores                                    │
│  • Explains reasoning for each recommendation                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. FastMCP Server Layer

**Purpose**: Exposes product search tools via Model Context Protocol

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

### 2. Tool Implementations

#### recommend_products Tool

```
Input:  "I want to buy a scarf for 40 dollars"
        ↓
    [SerpAPI Search]
        ↓
    [Parse Results]
        ↓
Output: Structured product data (JSON)
```

**Process**:
1. Receives natural language query
2. Constructs optimized search query for shopping
3. Queries SerpAPI for product listings
4. Parses raw results into structured format
5. Returns JSON with product names, prices, descriptions, URLs

**No AI reasoning at this level** - just data retrieval and structuring.

#### search_products Tool

```
Input:  "wireless headphones"
        ↓
    [SerpAPI Search]
        ↓
Output: Raw search results
```

**Process**:
1. Receives search query
2. Queries SerpAPI
3. Returns raw results for lower-level access

## Data Flow

### Request Flow

```
1. User asks AI agent for product recommendations
   ↓
2. AI agent calls recommend_products MCP tool
   ↓
3. MCP server validates parameters
   ↓
4. MCP server searches SerpAPI
   ↓
5. MCP server parses results into structured format
   ↓
6. MCP server returns JSON with product data
   ↓
7. AI agent receives product data
   ↓
8. AI agent analyzes products:
   - Evaluates match to user requirements
   - Considers budget constraints
   - Compares features and value
   - Generates reasoning for each product
   - Ranks products by suitability
   ↓
9. AI agent presents recommendations to user with reasoning
```

### Example Data Flow

**User Query:**
```
"I want to buy a scarf for 40 dollars"
```

**MCP Server Response:**
```json
{
  "query": "I want to buy a scarf for 40 dollars",
  "products": [
    {
      "name": "Winter Wool Scarf",
      "price": "$38.99",
      "description": "100% merino wool, various colors",
      "url": "https://example.com/product1"
    },
    {
      "name": "Cashmere Blend Scarf",
      "price": "$35.99",
      "description": "Soft cashmere blend, multiple patterns",
      "url": "https://example.com/product2"
    }
  ],
  "count": 2,
  "note": "These are search results. The agent should analyze and provide reasoning."
}
```

**AI Agent Analysis & Response to User:**
```
I found 2 great scarves within your $40 budget:

1. **Winter Wool Scarf - $38.99** ⭐ Score: 9/10
   This is an excellent choice. The 100% merino wool provides superior warmth 
   and quality. At $38.99, it's well within your budget and offers great value.
   Merino wool is naturally breathable and soft against skin.

2. **Cashmere Blend Scarf - $35.99** ⭐ Score: 8/10
   A more affordable luxury option. The cashmere blend provides softness and 
   warmth at a lower price point ($35.99). The multiple pattern options give 
   you style flexibility. Great value if you want a touch of cashmere without 
   the high price.

My top recommendation is the Winter Wool Scarf for its superior quality and warmth.
```

## Integration Points

### External APIs

#### SerpAPI
- **Usage**: Real-time product search across retailers
- **Calls per request**: 1
- **Authentication**: API key via environment variable
- **Results**: Aggregated product listings



## Error Handling

### Error Flow

```
Try:
    Validate input parameters
    ↓
    Execute SerpAPI search
    ↓
    Parse results
    ↓
    Return structured data
Except APIError:
    Log error
    Return {"error": "API error message"}
Except ValidationError:
    Log error
    Return {"error": "Validation error message"}
Except Exception:
    Log with traceback
    Return {"error": "Generic error message"}
```

### Error Types

1. **API Key Errors**: Missing or invalid SERPAPI_API_KEY
2. **API Quota Errors**: SerpAPI rate limits exceeded
3. **Network Errors**: Connection failures
4. **Parsing Errors**: Invalid search results format
5. **Validation Errors**: Invalid parameters (max_results bounds, etc.)

## Performance Characteristics

### Latency Breakdown

```
Total Request Time: ~2-4 seconds
    ├── Search Products: ~2-3 seconds (SerpAPI)
    └── Parse Results: ~0.5-1 second (local processing)
```

**Note**: The AI agent's analysis time is additional and depends on the agent's LLM speed.

### Optimization Strategies

1. **Efficient Parsing**: Lightweight result parsing
2. **Result Limiting**: max_results parameter (default 10, max 20 for recommend_products)
3. **Simple Processing**: No complex AI processing on server side
4. **Stateless Design**: No session management overhead

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
- External API rate limits (SerpAPI)
- Docker containerization for easy deployment
- Lightweight processing (no AI model loading)

## Security

### API Key Management

```
Environment Variables
    └── SERPAPI_API_KEY (required)
        • Never logged or exposed
        • Loaded from environment only
        • Validated at startup
```

### Input Validation

- Query length limits
- max_results bounds checking (10 default, 20 max for recommend_products, 100 max for search_products)
- Parameter type validation
- Error message sanitization

## Monitoring and Logging

### Log Levels

```
DEBUG:   Detailed execution flow
INFO:    Search queries and result counts
WARNING: API issues or parsing problems
ERROR:   Failures and exceptions
```

### Key Metrics

- Request count
- Response times
- SerpAPI call counts
- Error rates
- Search result quality

## Deployment Architecture

### Docker Deployment

```
Docker Container
    ├── Python 3.10+
    ├── uv package manager
    ├── Application code
    ├── Dependencies (FastMCP, LangChain Community, SerpAPI)
    └── Environment configuration

Exposed:
    └── Port 8000 (HTTP)

Environment:
    ├── SERPAPI_API_KEY (required)
    ├── HOST (default: 0.0.0.0)
    ├── PORT (default: 8000)
    ├── MCP_TRANSPORT (default: http)
    ├── MCP_JSON_RESPONSE (default: true)
    └── LOG_LEVEL (default: INFO)
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

## Agent Integration Guide

### Recommended System Prompt for AI Agents

When using this MCP server, configure your AI agent with a system prompt like:

```
When users ask for product recommendations:

1. Use the recommend_products tool to search for products based on their query
2. Analyze each returned product considering:
   - How well it matches the user's specific requirements
   - Whether the price fits within their stated budget
   - The value proposition (quality vs. price)
   - Any specific features or constraints mentioned
   - Availability and retailer reputation

3. For each relevant product, provide:
   - A clear explanation of why it's a good match
   - A recommendation score (1-10 scale)
   - Any important caveats or considerations
   - Comparison with other options

4. Rank products by how well they meet the user's needs
5. Present the top 3-5 recommendations with detailed reasoning
6. Provide direct purchase links for convenience

Always explain your reasoning transparently so users understand why you're 
recommending specific products.
```

### Example Agent Conversation Flow

```
User: "I need wireless headphones under $100"

Agent: [Calls recommend_products MCP tool]

MCP Server: [Returns 5 product results with names, prices, descriptions, URLs]

Agent: [Analyzes products and responds to user with]:
  "I found several great options for wireless headphones under $100:
   
   1. Sony WH-CH710N - $89.99 ⭐ 9/10
      [Detailed reasoning about features, value, fit to requirements]
   
   2. JBL Tune 760NC - $79.99 ⭐ 8/10
      [Reasoning]
   
   3. Anker Soundcore Q30 - $69.99 ⭐ 8/10
      [Reasoning]
   
   My top pick is the Sony WH-CH710N because [explanation]..."
```

## Future Enhancements

### Planned Features

1. **Caching Layer**: Redis for common searches
2. **Advanced Filtering**: Price ranges, categories, ratings in query parsing
3. **Multiple Providers**: Add more search providers beyond SerpAPI
4. **Structured Query Parameters**: Optional structured input format
5. **Result Enrichment**: Add more metadata (reviews, ratings, shipping info)
6. **Price Tracking**: Historical price data
7. **Image URLs**: Include product image URLs when available

### Architectural Improvements

1. **Async/Await**: Non-blocking SerpAPI calls
2. **Parallel Searches**: Query multiple providers simultaneously
3. **Enhanced Parsing**: Better extraction of product details
4. **Rate Limiting**: Built-in rate limiting for API protection
5. **Result Caching**: Cache recent searches to reduce API calls

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Protocol** | FastMCP | MCP server framework |
| **Search** | SerpAPI | Product search across retailers |
| **Integration** | SerpAPI Python Client | Direct SerpAPI integration |
| **Runtime** | Python 3.10+ | Application runtime |
| **Package Manager** | uv | Fast Python package management |
| **Containerization** | Docker | Deployment and isolation |

## Conclusion

The Shopping Agent represents a clean, lightweight MCP server implementation:
- ✅ Focused responsibility (data retrieval only)
- ✅ Fast response times (no LLM processing)
- ✅ Simple deployment and scaling
- ✅ Clear separation of concerns
- ✅ Agent-friendly architecture
- ✅ Production-ready error handling

The server does one thing well: searches for products and returns structured data.
Your AI agent does what it does best: intelligent analysis and recommendations.

Ready for deployment and real-world use! 🚀
