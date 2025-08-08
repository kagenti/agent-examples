import logging
import sys

from typing import Callable
from autogen import coding, ConversableAgent, register_function
from autogen.mcp.mcp_client import Toolkit

from slack_researcher.config import Settings
from slack_researcher.llm import LLMConfig
from slack_researcher.prompts import (
    ASSISTANT_PROMPT,
    GOAL_JUDGE_PROMPT,
    PLANNER_MESSAGE,
    REFLECTION_ASSISTANT_PROMPT,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s: %(message)s')


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

        # Generic LLM completion, used for servicing Open WebUI originated requests
        self.generic_assistant = ConversableAgent(
            name="Generic_Assistant",
            llm_config=llm_config.openai_llm_config,
            code_execution_config=False,
            human_input_mode="NEVER",
        )

        # The assistant agent is responsible for executing each step of the plan, including calling tools
        self.assistant = ConversableAgent(
            name="Research_Assistant",
            system_message=ASSISTANT_PROMPT,
            code_execution_config=False,
            llm_config=llm_config.openai_llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: "tool_response" not in msg
            and msg["content"] == "",
        )

        # Determines whether the ultimate objective has been met
        self.goal_judge = ConversableAgent(
            name="GoalJudge",
            system_message=GOAL_JUDGE_PROMPT,
            code_execution_config=False,
            llm_config=llm_config.openai_llm_config,
            human_input_mode="NEVER",
        )

        # Step Critic
        self.step_critic = ConversableAgent(
            name="Step_Critic",
            code_execution_config=False,
            llm_config=llm_config.openai_llm_config,
            human_input_mode="NEVER",
        )

        # Reflection Assistant: Reflect on plan progress and give the next step
        self.reflection_assistant = ConversableAgent(
            name="ReflectionAssistant",
            system_message=REFLECTION_ASSISTANT_PROMPT,
            code_execution_config=False,
            llm_config=llm_config.openai_llm_config,
            human_input_mode="NEVER",
        )

        # Report Generator
        self.report_generator = ConversableAgent(
            name="Report_Generator",
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
                "##ANSWER##" in msg["content"]
                or "## Answser" in msg["content"]
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
            mcp_toolkit.register_for_llm(self.assistant)
            tool_descriptions = []
            for tool in mcp_toolkit.tools:
                tool_descriptions.append({tool.name : tool.description})
            tool_descriptions = str(tool_descriptions)
            logging.info("Tool descriptions: %s", tool_descriptions)
        else:
            logging.info("No MCP tools to register")

        # Provides the initial high level plan
        self.planner = ConversableAgent(
            name="Planner",
            system_message=PLANNER_MESSAGE.format(tool_descriptions=tool_descriptions),
            llm_config=llm_config.planner_llm_config,
            human_input_mode="NEVER",
        )

