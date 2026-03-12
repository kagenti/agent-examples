import logging
import sys

from typing import Callable
from autogen import ConversableAgent, register_function
from autogen.mcp.mcp_client import Toolkit

from slack_researcher.config import Settings, settings
from slack_researcher.llm import LLMConfig
from slack_researcher.prompts import (
    ASSISTANT_PROMPT,
    REQUIREMENT_IDENTIFIER_PROMPT,
    CHANNEL_FILTER_PROMPT,
    SUMMARIZER_PROMPT,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL, stream=sys.stdout, format="%(levelname)s: %(message)s")


class Agents:
    def __init__(
        self,
        config: Settings = None,
        assistant_tools: dict[str, Callable] = None,
        mcp_toolkit: Toolkit = None,
    ):
        if not config:
            config = Settings()

        llm_config = LLMConfig(config)

        self.slack_channel_assistant = ConversableAgent(
            system_message=ASSISTANT_PROMPT,
            name="Slack_Channel_Assistant",
            llm_config=llm_config.openai_llm_config,
            code_execution_config=False,
            human_input_mode="NEVER",
        )

        self.intent_classifier = ConversableAgent(
            name="Intent_Classifier",
            llm_config=llm_config.intent_classifier_llm_config,
            code_execution_config=False,
            human_input_mode="NEVER",
        )

        self.requirement_identifier = ConversableAgent(
            name="Requirement_Identifier",
            system_message=REQUIREMENT_IDENTIFIER_PROMPT,
            llm_config=llm_config.user_requirement_llm_config,
            code_execution_config=False,
            human_input_mode="NEVER",
        )

        self.channel_assistant_no_tools = ConversableAgent(
            name="Slack_Channel_Assistant_NO_TOOLS",
            system_message=CHANNEL_FILTER_PROMPT,
            llm_config=llm_config.channel_llm_config,
            code_execution_config=False,
            human_input_mode="NEVER",
        )

        self.report_generator = ConversableAgent(
            name="Report_Generator",
            system_message=SUMMARIZER_PROMPT,
            llm_config=llm_config.openai_llm_config,
            code_execution_config=False,
            human_input_mode="NEVER",
        )

        # User Proxy chats with assistant on behalf of user and executes tools
        self.user_proxy = ConversableAgent(
            name="User",
            human_input_mode="NEVER",
            code_execution_config=False,
            is_termination_msg=lambda msg: msg
            and "content" in msg
            and msg["content"] is not None
            and (
                "##ANSWER" in msg["content"]
                or "## Answer" in msg["content"]
                or "##TERMINATE##" in msg["content"]
                or ("tool_calls" not in msg and msg["content"] == "")
            ),
        )

        tool_descriptions = ""

        if assistant_tools:
            for description, tool in assistant_tools.items():
                logging.info("Registering tool %s", description)
                register_function(
                    tool,
                    caller=self.assistant,
                    executor=self.user_proxy,
                    name=description,
                    description=description,
                )
                tool_descriptions += description + "\n\n"

        if mcp_toolkit is not None:
            logging.info("Registering MCP tool")
            logging.info(mcp_toolkit)
            mcp_toolkit.register_for_execution(self.user_proxy)
            mcp_toolkit.register_for_llm(self.slack_channel_assistant)
            tool_descriptions = []
            for tool in mcp_toolkit.tools:
                tool_descriptions.append({tool.name: tool.description})
            tool_descriptions = str(tool_descriptions)
            logging.info("Tool descriptions: %s", tool_descriptions)
        else:
            logging.info("No MCP tools to register")
