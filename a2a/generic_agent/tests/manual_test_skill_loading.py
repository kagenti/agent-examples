"""
Manual Test for Generic Agent with Summarizer Skill

This is a MANUAL TEST that requires a running Ollama instance and cannot be
run in automated CI/CD pipelines. It verifies that the generic agent can:
1. Load the summarizer skill from the SKILL_FOLDERS environment variable
2. Start successfully with Ollama as the LLM backend
3. Process a summarization request using the loaded skill
4. Generate a response that demonstrates the skill was used

Prerequisites:
- Ollama running locally on default port (11434)
- A model available (default: llama3.2:3b-instruct-fp16)
- The summarizer skill located at ../../../skills/summarizer (relative to this test file)

Usage:
    Run this test manually using the provided shell script:
    ./run_skill_test.sh
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the generic_agent module to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generic_agent.config import Configuration
from generic_agent.graph import get_graph, get_mcpclient, load_skills_content
from langchain_core.messages import HumanMessage


async def test_skill_loading():
    """Test that the summarizer skill is loaded and used by the agent."""

    print("=" * 80)
    print("Generic Agent - Summarizer Skill Test")
    print("=" * 80)

    # Step 1: Set up environment to load the summarizer skill
    # Use relative path from this test file to the skills directory
    test_dir = Path(__file__).parent
    repo_root = test_dir.parent.parent.parent
    skill_path = str(repo_root / "skills" / "summarizer")
    os.environ["SKILL_FOLDERS"] = skill_path

    print(f"\n✓ Set SKILL_FOLDERS environment variable to: {skill_path}")

    # Verify the skill folder exists
    if not Path(skill_path).exists():
        print(f"\n✗ ERROR: Skill folder does not exist at {skill_path}")
        return False
    print("✓ Verified skill folder exists")

    # Step 2: Load configuration
    config = Configuration()
    print("\n✓ Configuration loaded:")
    print(f"  - LLM Model: {config.LLM_MODEL}")
    print(f"  - LLM API Base: {config.LLM_API_BASE}")
    print(f"  - Skill Folders: {config.SKILL_FOLDERS}")

    # Step 3: Verify skill content is loaded
    print("\n" + "=" * 80)
    print("Testing Skill Loading")
    print("=" * 80)

    skills_content = load_skills_content()

    if not skills_content:
        print("\n✗ ERROR: No skills content loaded")
        return False

    print("\n✓ Skills content loaded successfully")
    print(f"  - Content length: {len(skills_content)} characters")

    # Verify the summarizer skill is in the content
    if "summarizer" not in skills_content.lower():
        print("\n✗ ERROR: Summarizer skill not found in loaded content")
        return False

    print("✓ Verified 'summarizer' skill is present in loaded content")

    # Check for key skill components
    skill_indicators = ["summarization", "bullet", "executive summary", "key points"]

    found_indicators = [ind for ind in skill_indicators if ind.lower() in skills_content.lower()]
    print(f"✓ Found {len(found_indicators)}/{len(skill_indicators)} skill indicators:")
    for indicator in found_indicators:
        print(f"  - {indicator}")

    # Step 4: Initialize MCP client (may be empty, that's OK)
    print("\n" + "=" * 80)
    print("Initializing Agent Components")
    print("=" * 80)

    try:
        mcpclient = get_mcpclient()
        print("\n✓ MCP client initialized")

        # Try to get tools (may be empty)
        try:
            tools = await mcpclient.get_tools()
            print(f"✓ MCP tools available: {len(tools)}")
        except Exception as e:
            print(f"⚠ No MCP tools available (this is OK for this test): {e}")
    except Exception as e:
        print(f"⚠ MCP client initialization warning: {e}")
        print("  (This is OK - agent will work without MCP tools)")

    # Step 5: Create the graph with the skill loaded
    print("\n✓ Creating agent graph with loaded skills...")

    try:
        graph = await get_graph(mcpclient)
        print("✓ Agent graph created successfully")
    except Exception as e:
        print(f"\n✗ ERROR: Failed to create agent graph: {e}")
        return False

    # Step 6: Test the agent with a summarization request
    print("\n" + "=" * 80)
    print("Testing Agent with Summarization Request")
    print("=" * 80)

    # Create a test document that needs summarization
    test_document = """
    The quarterly review meeting was held on January 15, 2026. The team discussed
    the product launch timeline and budget allocation. It was decided to move the
    launch date to March 1, 2026, to allow for additional testing. The marketing
    budget was approved at $250,000. Sarah will finalize the marketing plan by
    January 22. Mike needs to complete beta testing by February 1. The team agreed
    to review the pricing strategy in the next meeting. Revenue projections show
    a 23% increase over last quarter. Customer satisfaction scores improved to 4.5
    out of 5. The development team identified three critical bugs that must be
    fixed before launch. John will coordinate with the QA team to prioritize these
    issues. The meeting concluded with a commitment to weekly status updates.
    """

    user_prompt = f"""Please summarize the following meeting notes in bullet points:

{test_document}

Provide a clear, concise summary with the key decisions, action items, and important data points."""

    print(f"\n✓ Test prompt prepared (length: {len(user_prompt)} characters)")
    print("\nPrompt preview:")
    print("-" * 80)
    print(user_prompt[:200] + "...")
    print("-" * 80)

    # Create input message
    messages = [HumanMessage(content=user_prompt)]
    input_data = {"messages": messages}

    print("\n✓ Invoking agent...")
    print("  (This may take 30-60 seconds depending on the model)")

    try:
        # Stream the graph execution
        final_output = None
        step_count = 0

        async for event in graph.astream(input_data, stream_mode="updates"):
            step_count += 1
            print(f"\n  Step {step_count}:")
            for key, value in event.items():
                # Print abbreviated output
                value_str = str(value)
                if len(value_str) > 200:
                    value_str = value_str[:200] + "..."
                print(f"    {key}: {value_str}")
            final_output = event

        print(f"\n✓ Agent completed execution in {step_count} steps")

    except Exception as e:
        print(f"\n✗ ERROR: Agent execution failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Step 7: Verify the response
    print("\n" + "=" * 80)
    print("Analyzing Agent Response")
    print("=" * 80)

    if not final_output:
        print("\n✗ ERROR: No output received from agent")
        return False

    # Extract the final answer
    final_answer = final_output.get("assistant", {}).get("final_answer")

    if not final_answer:
        print("\n✗ ERROR: No final answer in agent output")
        print(f"Output keys: {final_output.keys()}")
        return False

    print(f"\n✓ Final answer received (length: {len(final_answer)} characters)")
    print("\nFinal Answer:")
    print("=" * 80)
    print(final_answer)
    print("=" * 80)

    # Step 8: Verify the response shows skill usage
    print("\n" + "=" * 80)
    print("Verifying Skill Usage")
    print("=" * 80)

    # Check for indicators that the summarization skill was used
    summary_indicators = [
        "•",  # Bullet points
        "-",  # Dashes for lists
        "summary",
        "key",
        "decision",
        "action",
        "budget",
        "march",
        "250",
    ]

    found_in_response = [ind for ind in summary_indicators if ind.lower() in final_answer.lower()]

    print(f"\n✓ Found {len(found_in_response)}/{len(summary_indicators)} summary indicators in response:")
    for indicator in found_in_response[:5]:  # Show first 5
        print(f"  - '{indicator}'")

    # Check if response is actually a summary (shorter than input)
    response_length = len(final_answer.split())
    input_length = len(test_document.split())
    compression_ratio = response_length / input_length

    print("\n✓ Compression analysis:")
    print(f"  - Input words: {input_length}")
    print(f"  - Response words: {response_length}")
    print(f"  - Compression ratio: {compression_ratio:.2f}")

    if compression_ratio > 1.5:
        print("  ⚠ Warning: Response is longer than input (may not be a proper summary)")
    else:
        print("  ✓ Response is appropriately condensed")

    # Final verdict
    print("\n" + "=" * 80)
    print("Test Results")
    print("=" * 80)

    success_criteria = [
        ("Skill folder exists", Path(skill_path).exists()),
        ("Skills content loaded", bool(skills_content)),
        ("Summarizer skill found", "summarizer" in skills_content.lower()),
        ("Agent graph created", graph is not None),
        ("Agent executed successfully", final_output is not None),
        ("Final answer received", bool(final_answer)),
        ("Summary indicators present", len(found_in_response) >= 3),
    ]

    print("\nSuccess Criteria:")
    all_passed = True
    for criterion, passed in success_criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {criterion}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ TEST PASSED: Summarizer skill loaded and used successfully!")
    else:
        print("✗ TEST FAILED: Some criteria not met")
    print("=" * 80)

    return all_passed


async def main():
    """Main test runner."""
    try:
        success = await test_skill_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

# Made with Bob
