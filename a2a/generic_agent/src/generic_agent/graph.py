import logging
from pathlib import Path
from typing import List

from langchain_core.messages import AIMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from generic_agent.config import Configuration

config = Configuration()
logger = logging.getLogger(__name__)


# Extend MessagesState to include a final answer
class ExtendedMessagesState(MessagesState):
    final_answer: str = ""


def _get_mcp_urls() -> List[str]:
    """Helper function to parse MCP URLs from environment variable."""
    urls_str = config.MCP_URLS
    return [url.strip() for url in urls_str.split(",") if url.strip()]


def get_mcpclient() -> MultiServerMCPClient:
    """
    Create MCP client with error handling.

    If individual MCP servers fail to connect, logs warnings but continues
    with available servers.

    Returns:
        MultiServerMCPClient instance (may have empty configs if all servers fail)

    Note:
        This function is not cached to allow retry on transient failures.
        Each call creates a new client instance.
    """
    urls = _get_mcp_urls()

    client_configs = {}
    transport = config.MCP_TRANSPORT

    for i, url in enumerate(urls, 1):
        client_configs[f"mcp{i}"] = {
            "url": url,
            "transport": transport,
        }

    if not client_configs:
        logger.warning("No MCP servers configured successfully. Agent will work without MCP tools.")

    return MultiServerMCPClient(client_configs)


def get_mcp_server_names() -> List[str]:
    """
    Extract MCP server names from URLs.

    Strips protocol (http/https), port, and path to get just the host names.
    Example: "http://weather-tool:8000/mcp" -> "weather-tool"

    Returns:
        List of MCP server host names
    """
    urls = _get_mcp_urls()

    mcp_names = []
    for url in urls:
        # Remove protocol
        name = url.replace("http://", "").replace("https://", "")
        # Remove port and path (everything after first :)
        name = name.split(":")[0]
        # Remove /mcp or any path
        name = name.split("/")[0]
        if name:
            mcp_names.append(name)

    return mcp_names


def get_skill_folder_paths() -> List[str]:
    """
    Extract skill folder paths from the SKILL_FOLDERS configuration.

    The SKILL_FOLDERS field contains comma-separated folder paths.
    Example: "/app/skills/pdf,/app/skills/skill-creator" -> ["/app/skills/pdf", "/app/skills/skill-creator"]

    Returns:
        List of skill folder paths
    """
    folders_str = config.SKILL_FOLDERS
    if not folders_str or not folders_str.strip():
        return []

    folder_paths = []
    for folder in folders_str.split(","):
        folder = folder.strip()
        if folder:
            folder_paths.append(folder)

    return folder_paths


def load_skills_content() -> str:
    """
    Load skill content from skill folders with error handling.

    Reads all relevant files from each skill folder:
    - SKILL.md: Main skill description and instructions
    - *.py: Python scripts and tools
    - *.md: Additional documentation files

    If individual skills fail to load, logs warnings but continues with other skills.

    Returns:
        Combined skill content as a string, or empty string if no skills found

    Note:
        Content is limited to 100KB total to avoid exceeding LLM context windows
        and ConfigMap size limits (1 MiB). A warning is logged if this limit is exceeded.
    """
    skill_folders = get_skill_folder_paths()
    if not skill_folders:
        return ""

    skills_content = []
    failed_skills = []
    total_size = 0
    MAX_CONTENT_SIZE = 100 * 1024  # 100KB limit (summarizer skill is ~45KB)

    for folder_path in skill_folders:
        try:
            skill_path = Path(folder_path)
            if not skill_path.exists() or not skill_path.is_dir():
                logger.warning(f"Skill folder does not exist or is not a directory: {folder_path}")
                failed_skills.append(skill_path.name if skill_path else folder_path)
                continue

            skill_name = skill_path.name
            skill_parts = [f"### Skill: {skill_name}\n"]

            # Load SKILL.md first (main description)
            skill_md_path = skill_path / "SKILL.md"
            if skill_md_path.exists() and skill_md_path.is_file():
                try:
                    with open(skill_md_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            skill_parts.append(f"#### Main Description\n\n{content}")
                except Exception as e:
                    logger.warning(f"Failed to read {skill_md_path}: {e}")

            # Load Python scripts (these are tools/utilities the skill provides)
            python_files = sorted(skill_path.rglob("*.py"))
            if python_files:
                scripts_content = []
                for py_file in python_files:
                    try:
                        # Get relative path from skill folder
                        rel_path = py_file.relative_to(skill_path)
                        with open(py_file, "r", encoding="utf-8") as f:
                            code = f.read().strip()
                            if code:
                                content_piece = f"**File: {rel_path}**\n```python\n{code}\n```"
                                # Check size before adding
                                if total_size + len(content_piece) > MAX_CONTENT_SIZE:
                                    logger.warning(
                                        f"Skill content size limit ({MAX_CONTENT_SIZE} bytes) exceeded. "
                                        f"Skipping remaining files in {skill_name}."
                                    )
                                    break
                                scripts_content.append(content_piece)
                                total_size += len(content_piece)
                    except Exception as e:
                        logger.warning(f"Failed to read {py_file}: {e}")

                if scripts_content:
                    skill_parts.append("#### Available Scripts\n\n" + "\n\n".join(scripts_content))

            # Load additional markdown files (documentation, examples, etc.)
            md_files = sorted([f for f in skill_path.rglob("*.md") if f.name != "SKILL.md"])
            if md_files:
                docs_content = []
                for md_file in md_files:
                    try:
                        rel_path = md_file.relative_to(skill_path)
                        with open(md_file, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:
                                docs_content.append(f"**{rel_path}**\n\n{content}")
                    except Exception as e:
                        logger.warning(f"Failed to read {md_file}: {e}")

                if docs_content:
                    skill_parts.append("#### Additional Documentation\n\n" + "\n\n".join(docs_content))

            if len(skill_parts) > 1:  # More than just the header
                skills_content.append("\n\n".join(skill_parts))
        except Exception as e:
            logger.warning(f"Failed to load skill from {folder_path}: {e}")
            failed_skills.append(Path(folder_path).name)
            continue

    if failed_skills:
        logger.warning(f"Failed to load {len(failed_skills)} skill(s): {', '.join(failed_skills)}")

    if skills_content:
        loaded_count = len(skills_content)
        final_content = "\n\n" + "\n\n---\n\n".join(skills_content)
        logger.info(f"Successfully loaded {loaded_count} skill(s), total size: {len(final_content)} bytes")

        if len(final_content) > MAX_CONTENT_SIZE:
            logger.warning(
                f"Total skill content size ({len(final_content)} bytes) exceeds recommended limit "
                f"({MAX_CONTENT_SIZE} bytes). This may impact LLM context window and ConfigMap limits."
            )

        return final_content

    logger.info("No skills loaded")
    return ""


async def get_graph(client: MultiServerMCPClient) -> StateGraph:
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_API_BASE,
        temperature=0,
    )

    # Get tools asynchronously with error handling
    try:
        tools = await client.get_tools()
        if tools:
            logger.info(f"Successfully loaded {len(tools)} MCP tool(s)")
        else:
            logger.warning("No MCP tools available")
        llm_with_tools = llm.bind_tools(tools)
    except Exception as e:
        logger.warning(f"Failed to load MCP tools: {e}. Agent will work without MCP tools.")
        tools = []
        llm_with_tools = llm

    # Load skills content if available
    skills_content = load_skills_content()

    # Build system message
    base_instruction = "You are the **Generic Assistant**, a multi-purpose, tool-based expert. Your primary directive is to fulfill user requests by effectively utilizing the available **MCP tools**. You will select the most appropriate tool(s) based on the user's need (e.g., weather, calculations, data retrieval) and strictly adhere to their output to generate your final answer. Be precise and concise."

    if skills_content:
        system_content = f"{base_instruction}\n\n## Available Skills\n\nYou have access to the following specialized skills. Use them when the user's request matches the skill's capabilities:{skills_content}"
    else:
        system_content = base_instruction

    sys_msg = SystemMessage(content=system_content)

    # Node
    def assistant(state: ExtendedMessagesState) -> ExtendedMessagesState:
        result = llm_with_tools.invoke([sys_msg] + state["messages"])

        updated_state = {"messages": state["messages"] + [result]}

        # Set final_answer when LLM returns a text response (not a tool call)
        # This indicates the assistant has completed its reasoning and tool usage
        if isinstance(result, AIMessage) and not result.tool_calls:
            updated_state["final_answer"] = result.content

        return updated_state

    # Build graph
    builder = StateGraph(ExtendedMessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    graph = builder.compile()
    return graph
