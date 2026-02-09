"""Generalist agent using AG2 with MCP tools."""

import logging
import os
from typing import Any, Callable
from autogen import ConversableAgent, UserProxyAgent
from autogen.mcp.mcp_client import Toolkit
from simple_generalist.config import Settings
from simple_generalist.agent.prompts import GENERAL_AGENT_PROMPT

logger = logging.getLogger(__name__)


# OTEL Tracing config
project_name = "simple_generalist_agent"
if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
    # OTLP is configured → enable tracing only
    os.environ.setdefault("OTEL_TRACES_EXPORTER", "otlp")
    os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
    os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")

    # service identity
    project_name = "simple_generalist_agent"
    resource_parts = [
        f"service.name={project_name}",
        f"openinference.project.name={project_name}",
    ]
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(resource_parts)

    # Disable OpenLIT metrics explicitly
    os.environ.setdefault("OPENLIT_DISABLE_METRICS", "true")

    import openlit

    openlit.init(
        application_name=project_name,
        disable_metrics=True,
    )



class GeneralistAgent:
    """
    Generalist agent that uses AG2 for LLM interaction and MCP tools for actions.
    
    - Maintains conversation state
    - Calls LLM for next action
    - Executes tools via MCP
    - Iterates until completion or limits
    """
    
    def __init__(
        self,
        settings: Settings,
        mcp_toolkit: Toolkit | None = None,
        event_callback: Callable[[str, bool], Any] | None = None,
    ):
        """
        Initialize the generalist agent.
        
        Args:
            settings: Application settings
            mcp_toolkit: Optional AG2 MCP toolkit with connected servers
            event_callback: Optional callback for progress events (message, is_final)
        """
        self.settings = settings
        self.mcp_toolkit = mcp_toolkit
        self.event_callback = event_callback
        
        # Initialize AG2 agent 
        self._init_ag2_agent()
    
    def _init_ag2_agent(self):
        """Initialize the AG2 conversable agent without registering tools."""
        # Build LLM config
        llm_config = {
            "api_type": "openai",
            "model": self.settings.LLM_MODEL,
            "temperature": self.settings.LLM_TEMPERATURE,
            # Don't set max_tokens - let AG2 calculate it based on model context window
            # Setting it explicitly can cause issues with large prompts (many tools)
        }
        # Add API key if provided
        if self.settings.LLM_API_KEY:
            llm_config["api_key"] = self.settings.LLM_API_KEY
        
        # Add base URL if provided (for custom endpoints)
        if self.settings.LLM_BASE_URL:
            llm_config["base_url"] = self.settings.LLM_BASE_URL

        if self.settings.EXTRA_HEADERS:
            llm_config["default_headers"] = self.settings.EXTRA_HEADERS
        
        system_message = GENERAL_AGENT_PROMPT.format(max_steps=self.settings.MAX_ITERATIONS)
        
        # Create the agent
        self.agent = ConversableAgent(
            name="generalist_agent",
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        if self.mcp_toolkit:
            self.mcp_toolkit.register_for_execution(self.user_proxy)

        # Register MCP tools with AG2 agent if toolkit is provided
        if self.mcp_toolkit:
            tool_count = len(self.mcp_toolkit.tools)
            self.mcp_toolkit.register_for_llm(self.agent)
            logger.info(f"Initialized AG2 agent with {tool_count} MCP tools")
        else:
            logger.warning("Initialized AG2 agent without MCP tools")
    
    
    async def _emit_event(self, message: str, final: bool = False):
        """Emit a progress event if callback is set."""
        if self.event_callback:
            try:
                await self.event_callback(message, final)
            except Exception as exc:
                logger.error(f"Error in event callback: {exc}")
    
    async def run_task(self, instruction: str) -> dict[str, Any]:
        """
        Run a task with the given instruction.
        
        Uses AG2's built-in conversation flow with MCP tools.
        
        Args:
            instruction: User instruction/query
            
        Returns:
            Dictionary with:
                - answer: Final answer text
                - iterations: Number of iterations
        """
        logger.info(f"Starting task: {instruction}")
        await self._emit_event("🤖 Starting task execution...")
        
        try:
            # Initiate chat - AG2 handles the tool calling loop
            await self._emit_event(f"🔄 Processing with AG2 agent...")
            
            # Run the synchronous initiate_chat in a thread pool to avoid blocking
            await self.user_proxy.a_initiate_chat(
                self.agent,
                message=instruction,
                max_turns=self.settings.MAX_ITERATIONS)
            
            # Get the final response from chat history
            chat_history = self.user_proxy.chat_messages.get(self.agent, [])
            
            # Extract final answer from the last assistant message
            final_answer = "No response generated"
            for msg in reversed(chat_history):
                if msg.get("role") == "assistant" and msg.get("content"):
                    final_answer = msg["content"]
                    break
            
            logger.info("Task completed successfully")
            
            result = {
                "answer": final_answer,
                "iterations": len(chat_history),
            }
            
            return result
            
        except Exception as exc:
            logger.error(f"Error during task execution: {exc}", exc_info=True)
            return {
                "answer": f"Error: {str(exc)}",
                "iterations": 0,
            }

# Made with Bob
