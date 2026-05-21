#!/bin/bash
# Helper script to set up your API keys
# 
# Usage:
#   1. Run this script once: ./setup_env.sh
#   2. It will create a .env.local file for you to edit
#   3. Then run: source .env.local && python shopping_agent.py

echo "🛍️  Shopping Agent Environment Setup"
echo "====================================="
echo ""

if [ -f ".env.local" ]; then
    echo "ℹ️  .env.local already exists"
    echo ""
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing .env.local"
        exit 0
    fi
fi

echo "Creating .env.local template..."
cat > .env.local << 'EOF'
# Shopping Agent MCP Server - Local Environment Configuration
# Load this file: source .env.local

# Required: OpenAI API Key (https://platform.openai.com/api-keys)
export OPENAI_API_KEY="your-openai-api-key-here"

# Required: SerpAPI Key (https://serpapi.com/manage-api-key)
export SERPAPI_API_KEY="your-serpapi-api-key-here"

# Optional: Server Configuration
export HOST="0.0.0.0"
export PORT="8000"
export MCP_TRANSPORT="http"
export LOG_LEVEL="INFO"
EOF

echo "✓ Created .env.local"
echo ""
echo "📝 Next steps:"
echo "   1. Edit .env.local and add your actual API keys"
echo "   2. Load the environment: source .env.local"
echo "   3. Run the server: python3 shopping_agent.py"
echo "   4. Test the server: python3 test_client.py (in another terminal)"

