import json
import logging
import sys
from typing import Callable
from autogen.mcp.mcp_client import Toolkit
from slack_researcher.agents import Agents
from slack_researcher.config import settings, Settings
from slack_researcher.data_types import ChannelInfo, ChannelList, UserIntent, UserRequirement
from slack_researcher.event import Event


logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL, stream=sys.stdout, format="%(levelname)s: %(message)s")


class SlackAgent:
    def __init__(
        self,
        config: Settings,
        eventer: Event = None,
        assistant_tools: dict[str, Callable] = None,
        mcp_toolkit: Toolkit = None,
        logger=None,
    ):
        self.agents = Agents(settings, assistant_tools, mcp_toolkit)
        self.eventer = eventer
        self.logger = logger or logging.getLogger(__name__)

        # State
        self.user_intent: UserIntent = None
        self.requirements: UserRequirement = None
        self.all_channels: str = ""
        self.relevant_channels: ChannelList = None
        self.channel_outputs = []

    async def _send_event(self, message: str, final: bool = False):
        self.logger.info(message)
        if self.eventer:
            await self.eventer.emit_event(message, final)
        else:
            self.logger.warning("No event handler registered")

    async def execute(self, user_query):
        self.user_query = self.extract_user_input(user_query)
        await self.classify_intent()
        await self.list_all_channels()
        await self.identify_requirements()
        await self.get_relevant_channels()

        if self.user_intent.intent == "LIST_CHANNELS":
            return await self.summarize_data(str(self.relevant_channels))

        await self.query_channels()
        return await self.summarize_data(str(self.channel_outputs))

    def extract_user_input(self, body):
        content = body[-1]["content"]
        latest_content = ""

        if isinstance(content, str):
            latest_content = content
        else:
            for item in content:
                if item["type"] == "text":
                    latest_content += item["text"]
                else:
                    self.logger.warning(f"Ignoring content with type {item['type']}")

        return latest_content

    async def classify_intent(self):
        prompt = f"Classify the intent of the user as either simply needing to list slack channel information or if their intent is querying the content of slack channels themselves. User query: {self.user_query}"
        response = await self.agents.user_proxy.a_initiate_chat(
            message=prompt, recipient=self.agents.intent_classifier, max_turns=1
        )
        self.user_intent = UserIntent(**json.loads(response.chat_history[-1]["content"]))
        await self._send_event(f"🧐 Identified user intent: {self.user_intent.intent}")

    async def identify_requirements(self):
        response = await self.agents.user_proxy.a_initiate_chat(
            message=self.user_query, recipient=self.agents.requirement_identifier, max_turns=1
        )
        self.requirements = UserRequirement(**json.loads(response.chat_history[-1]["content"]))
        await self._send_event(
            f"📇 Identified channel requirements. Channel names: {self.requirements.specific_channel_names}, Channel types: {self.requirements.types_of_channels}"
        )

    async def list_all_channels(self):
        await self._send_event("🔎 Fetching all channels")
        response = await self.agents.user_proxy.a_initiate_chat(
            message="Retrieve all slack channels that are found on my slack server. Use the slack tool to find it.",
            recipient=self.agents.slack_channel_assistant,
            max_turns=3,
        )
        for item in response.chat_history:
            if item.get("tool_responses"):
                for tool_response in item["tool_responses"]:
                    self.all_channels += tool_response.get("content")
        return response

    async def get_relevant_channels(self):
        self._send_event("👀 Identifying relevant channels")
        prompt = ""
        if self.requirements.specific_channel_names:
            prompt += f"User is looking for channels with specific names: {self.requirements.specific_channel_names}"
            if self.requirements.types_of_channels:
                prompt += f"\n User is also looking for channels of any name that meet the following criteria: {self.requirements.types_of_channels}"
        else:
            prompt += f"User is looking for channels of any name that meet the following criteria: {self.requirements.types_of_channels}"
        prompt += f"\n The list of slack channels is as follows: {self.all_channels}"

        response = await self.agents.user_proxy.a_initiate_chat(
            message=prompt, recipient=self.agents.channel_assistant_no_tools, max_turns=1
        )
        self.relevant_channels = ChannelList(**json.loads(response.chat_history[-1]["content"]))

        channel_names = [channel.name for channel in self.relevant_channels.channels]
        await self._send_event(
            f"🎯 Relevant channels identified: {channel_names}. Reason: {self.relevant_channels.explanation}"
        )

    async def query_channel(self, channel: ChannelInfo):
        await self._send_event(f"📖 Querying channel {channel.name}")
        prompt = f'Retrieve the history from the slack channel with ID "{channel.id}" using the Slack tool available to you. The data retrieved will be used to answer the following user query/instruction: {self.user_query}'
        response = await self.agents.user_proxy.a_initiate_chat(
            message=prompt, recipient=self.agents.slack_channel_assistant, max_turns=3
        )

        # We're going to capture the raw channel data for analysis later
        channel_data = ""
        for item in response.chat_history:
            if item.get("tool_responses"):
                for tool_response in item["tool_responses"]:
                    channel_data += tool_response.get("content")
        # If no tool output exists, just take the agent's response
        if channel_data == "":
            channel_data = response.chat_history[-1]["content"]
        data = {"channel_name": channel.name, "channel_id": channel.id, "output": channel_data}
        return data

    async def query_channels(self):
        for channel in self.relevant_channels.channels:
            self.channel_outputs.append(await self.query_channel(channel))

    async def summarize_data(self, data_to_summarize):
        await self._send_event("📄 Generating a final report")
        prompt = f"User query: {self.user_query}. \n Information gathered: {data_to_summarize}"
        response = await self.agents.user_proxy.a_initiate_chat(
            message=prompt, recipient=self.agents.report_generator, max_turns=1
        )
        return response.chat_history[-1]["content"]
