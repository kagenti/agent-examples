# Generic Agent - Skill Loading Test

This test verifies that the generic agent can successfully load and use the summarizer skill.

## Test Overview

The test (`test_skill_loading.py`) performs the following checks:

1. **Environment Setup**: Sets the `SKILL_FOLDERS` environment variable to point to the summarizer skill
2. **Skill Loading**: Verifies that the skill content is loaded from the specified folder
3. **Agent Initialization**: Creates the agent graph with the loaded skill
4. **Skill Usage**: Sends a summarization request and verifies the agent uses the skill
5. **Response Validation**: Checks that the response demonstrates proper summarization

## Prerequisites

### 1. Ollama Setup

You need Ollama running locally with a compatible model:

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai

# Pull a model (default: llama3.2:3b-instruct-fp16)
ollama pull llama3.2:3b-instruct-fp16

# Or use a different model by setting LLM_MODEL environment variable
# ollama pull llama3.1
# export LLM_MODEL="llama3.1"

# Start Ollama (usually runs automatically)
# Default endpoint: http://localhost:11434
```

### 2. Python Dependencies

Install the required dependencies:

```bash
# Navigate to the generic_agent directory
cd a2a/generic_agent

# Install dependencies using pip
pip install -e .

# Or using uv (if available)
uv pip install -e .
```

### 3. Verify Skill Location

Ensure the summarizer skill exists at the expected location:

```bash
# From the repository root
ls -la skills/summarizer

# Or from the generic_agent directory
ls -la ../../skills/summarizer
```

You should see:
- `SKILL.md` - Main skill documentation
- `README.md` - Skill readme
- `scripts/` - Python helper scripts

## Running the Test

### Basic Test Run

```bash
# Navigate to the tests directory
cd a2a/generic_agent/tests

# Run the test with automated checks
./run_skill_test.sh

# Or run directly with Python
python test_skill_loading.py
```

### With Custom Configuration

You can override the default configuration using environment variables:

```bash
# Use a different model
export LLM_MODEL="llama3.1"

# Use a different Ollama endpoint
export LLM_API_BASE="http://localhost:11434/v1"

# Use a different skill folder
export SKILL_FOLDERS="/path/to/your/skills/summarizer"

# Run the test
cd a2a/generic_agent/tests
./run_skill_test.sh
```

### Expected Output

A successful test run will show:

```
================================================================================
Generic Agent - Summarizer Skill Test
================================================================================

✓ Set SKILL_FOLDERS environment variable to: <repo_root>/skills/summarizer
✓ Verified skill folder exists

✓ Configuration loaded:
  - LLM Model: llama3.2:3b-instruct-fp16
  - LLM API Base: http://localhost:11434/v1
  - Skill Folders: <repo_root>/skills/summarizer

================================================================================
Testing Skill Loading
================================================================================

✓ Skills content loaded successfully
  - Content length: XXXXX characters
✓ Verified 'summarizer' skill is present in loaded content
✓ Found 4/4 skill indicators:
  - summarization
  - bullet
  - executive summary
  - key points

================================================================================
Initializing Agent Components
================================================================================

✓ MCP client initialized
⚠ No MCP tools available (this is OK for this test)

✓ Creating agent graph with loaded skills...
✓ Agent graph created successfully

================================================================================
Testing Agent with Summarization Request
================================================================================

✓ Test prompt prepared (length: XXX characters)
✓ Invoking agent...
  (This may take 30-60 seconds depending on the model)

  Step 1:
    assistant: ...
  Step 2:
    assistant: ...

✓ Agent completed execution in X steps

================================================================================
Analyzing Agent Response
================================================================================

✓ Final answer received (length: XXX characters)

Final Answer:
================================================================================
[Summary output here]
================================================================================

================================================================================
Verifying Skill Usage
================================================================================

✓ Found X/9 summary indicators in response:
  - '•'
  - 'summary'
  - 'key'
  - 'decision'
  - 'action'

✓ Compression analysis:
  - Input words: 150
  - Response words: 80
  - Compression ratio: 0.53
  ✓ Response is appropriately condensed

================================================================================
Test Results
================================================================================

Success Criteria:
  ✓ PASS: Skill folder exists
  ✓ PASS: Skills content loaded
  ✓ PASS: Summarizer skill found
  ✓ PASS: Agent graph created
  ✓ PASS: Agent executed successfully
  ✓ PASS: Final answer received
  ✓ PASS: Summary indicators present

================================================================================
✓ TEST PASSED: Summarizer skill loaded and used successfully!
================================================================================
```

## Troubleshooting

### Error: "Skill folder does not exist"

**Solution**: Verify the skill path is correct:
```bash
# From repository root
ls -la skills/summarizer

# Or from generic_agent directory
ls -la ../../skills/summarizer
```

### Error: "Failed to create agent graph"

**Possible causes**:
1. Ollama is not running
2. Model is not available
3. Network connectivity issues

**Solutions**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Verify model is available
ollama list

# Pull the model if missing
ollama pull llama3.2:3b-instruct-fp16
```

### Error: "No skills content loaded"

**Possible causes**:
1. SKILL.md file is missing
2. File permissions issue
3. Incorrect path

**Solutions**:
```bash
# Check file exists and is readable (from repository root)
cat skills/summarizer/SKILL.md

# Check permissions
ls -la skills/summarizer/
```

### Warning: "Response is longer than input"

This warning indicates the LLM may not have properly summarized the content. This can happen if:
- The model is too small or not well-suited for summarization
- The prompt needs adjustment
- The skill instructions need refinement

This is not a test failure, but indicates the quality of summarization could be improved.

## Test Configuration

The test uses the following default configuration:

- **LLM Model**: `llama3.2:3b-instruct-fp16`
- **LLM API Base**: `http://localhost:11434/v1`
- **Skill Path**: `<repo_root>/skills/summarizer` (computed relative to test file location)
- **MCP URLs**: `http://localhost:8000/mcp` (optional, not required for this test)

All of these can be overridden using environment variables.

## What the Test Validates

1. ✓ Skill folder exists and is accessible
2. ✓ Skill content (SKILL.md, scripts) is loaded correctly
3. ✓ Agent graph is created with the skill integrated
4. ✓ Agent can process requests using the loaded skill
5. ✓ LLM generates responses that demonstrate skill usage
6. ✓ Response shows characteristics of proper summarization

## Next Steps

After a successful test run, you can:

1. **Add more skills**: Set `SKILL_FOLDERS` to multiple comma-separated paths
2. **Test with MCP tools**: Start MCP servers and set `MCP_URLS`
3. **Use different models**: Try different Ollama models
4. **Integrate into applications**: Use the generic agent in your A2A workflows

## Related Files

- [`test_skill_loading.py`](./test_skill_loading.py) - The test script
- [`run_skill_test.sh`](./run_skill_test.sh) - Convenience script with pre-flight checks
- [`../src/generic_agent/graph.py`](../src/generic_agent/graph.py) - Graph implementation with skill loading
- [`../src/generic_agent/config.py`](../src/generic_agent/config.py) - Configuration settings
- [`../../../skills/summarizer/SKILL.md`](../../../skills/summarizer/SKILL.md) - Summarizer skill documentation

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the generic agent README
3. Check Ollama documentation: https://ollama.ai
4. Review the skill documentation in `skills/summarizer/SKILL.md`
