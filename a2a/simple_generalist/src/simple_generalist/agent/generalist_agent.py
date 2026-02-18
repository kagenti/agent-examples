"""Generalist agent using AG2 with MCP tools."""

import logging
import os
from typing import Any, Callable
from autogen import ConversableAgent, UserProxyAgent
from autogen.mcp.mcp_client import Toolkit
from simple_generalist.config import Settings
from simple_generalist.agent.prompts import GENERAL_AGENT_PROMPT

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from autogen.opentelemetry import instrument_llm_wrapper, instrument_agent

logger = logging.getLogger(__name__)

_SERVICE_NAME = "simple_generalist_agent"

# Map agent names → stable IDs
_AGENT_IDS: dict[str, str] = {
    "generalist_agent": "generalist-agent-001",
    "user_proxy": "user-proxy-001",
}


class AgentIdSpanProcessor(SpanProcessor):
    """Injects gen_ai.agent.id on any span that carries gen_ai.agent.name.

    AG2's instrumentation sets gen_ai.agent.name but not gen_ai.agent.id.
    This processor maps agent names to stable IDs so downstream consumers
    (e.g. kagenti-compatible backends) see the field they expect.
    """

    def __init__(self, agent_ids: dict[str, str]) -> None:
        self._agent_ids = agent_ids

    def on_start(self, span: ReadableSpan, parent_context=None) -> None:
        agent_name = span.attributes.get("gen_ai.agent.name") if span.attributes else None
        if agent_name and agent_name in self._agent_ids:
            span.set_attribute("gen_ai.agent.id", self._agent_ids[agent_name])

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


_tracer_provider: TracerProvider | None = None
_tracing_initialized = False


def _init_tracing() -> TracerProvider:
    """Initialize the OpenTelemetry TracerProvider and instrument LLM calls.

    Safe to call multiple times — only the first call has any effect.
    """
    global _tracer_provider, _tracing_initialized
    if _tracing_initialized:
        return _tracer_provider  # type: ignore[return-value]

    resource = Resource.create(attributes={"service.name": _SERVICE_NAME})
    _tracer_provider = TracerProvider(resource=resource)
    _tracer_provider.add_span_processor(AgentIdSpanProcessor(_AGENT_IDS))

    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        _tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]))
        )
        logger.info("AG2 OpenTelemetry tracing enabled (OTLP endpoint: %s)", os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"])
    elif os.environ.get("OTEL_CONSOLE_TRACING", "").lower() in ("true", "1", "yes"):
        _tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        logger.info("AG2 OpenTelemetry tracing enabled (console exporter)")

    trace.set_tracer_provider(_tracer_provider)
    instrument_llm_wrapper(tracer_provider=_tracer_provider)

    _tracing_initialized = True
    return _tracer_provider




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
        self._tracer_provider = _init_tracing()

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

        # Instrument agents for tracing
        instrument_agent(self.agent, tracer_provider=self._tracer_provider)
        instrument_agent(self.user_proxy, tracer_provider=self._tracer_provider)
    
    
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
                - error: True if the task failed
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
                "error": False,
            }

            return result

        except Exception as exc:
            logger.error(f"Error during task execution: {exc}", exc_info=True)
            return {
                "answer": f"Error: {str(exc)}",
                "iterations": 0,
                "error": True,
            }

# Made with Bob
