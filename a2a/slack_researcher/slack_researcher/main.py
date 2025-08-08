import logging
import json
import asyncio
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Any

from autogen.mcp.mcp_client import Toolkit
from slack_researcher.event import Event
from slack_researcher.agents import Agents
from slack_researcher.config import Settings
from slack_researcher.prompts import STEP_CRITIC_PROMPT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s: %(message)s')

class PlanExecutionError(Exception):
    pass


@dataclass
class PlanContext:
    step_index: int = 0
    plan_dict: dict = field(default_factory=dict)
    latest_content: str = ""
    image_descriptions: List[str] = field(default_factory=list)
    answer_output: List[Any] = field(default_factory=list)
    steps_taken: List[str] = field(default_factory=list)
    last_step: str = ""
    last_output: Any = ""


class RagAgent:

    def __init__(
        self,
        config: Settings,
        eventer: Event,
        assistant_tools: dict[str, Callable],
        mcp_toolkit: Toolkit,
        logger=None,
    ):
        self.eventer = eventer
        self.config = config
        self.agents = Agents(
            self.config, assistant_tools=assistant_tools, mcp_toolkit=mcp_toolkit
        )
        self.logger = logger or logging.getLogger(__name__)
        self.context = PlanContext()

    async def run_workflow(self, body: list[dict]):
        # Parse instructions from user
        plan_instruction = self._extract_user_input(body)

        # Create an initial plan with the instructions
        self.context.plan_dict = await self._generate_plan(plan_instruction)

        # Plan should be a list. If not, return back to user
        if isinstance(self.context.plan_dict, str):
            return self.context.plan_dict

        # Execute plan
        final_answer = await self._execute_plan()
        return final_answer

    def _extract_user_input(self, body):
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

    async def _generate_plan(self, instruction):
        await self.eventer.emit_event(message="Creating a plan...")
        try:
            response = await self.agents.user_proxy.a_initiate_chat(
                message=instruction, max_turns=1, recipient=self.agents.planner
            )
            print(response)
            return json.loads(response.chat_history[-1]["content"])
        except Exception as e:
            self.logger.exception("Plan generation failed")
            return f"Unable to assemble a plan. Error: {e}"

    async def _execute_plan(self):
        for self.context.step_index in range(self.config.MAX_PLAN_STEPS):
            instruction = await self._determine_next_instruction()

            if not instruction or "##TERMINATE##" in instruction:
                break

            self.context.last_output = await self._execute_instruction(instruction)
            self.context.last_step = instruction

        return await self._summarize_results()

    async def _determine_next_instruction(self):
        if self.context.step_index == 0:
            return self.context.plan_dict["steps"][0]

        await self.eventer.emit_event(message="Planning the next step...")

        # First check if the previous step was successful
        output = await self.agents.user_proxy.a_initiate_chat(
            recipient=self.agents.step_critic,
            max_turns=1,
            message=STEP_CRITIC_PROMPT.format(
                last_step=self.context.last_step,
                context=self.context.answer_output,
                last_output=self.context.last_output,
            ),
        )
        was_job_accomplished = output.chat_history[-1]["content"]
        reflection_message = self.context.last_step

        # Only store the output of the last step to the context if it was successful
        # Throw away output of unsucessful steps
        if "##NO##" in was_job_accomplished:
            reflection_message = f"The previous step was {self.context.last_step} but was not accomplished: {was_job_accomplished}."
        else:
            self.context.answer_output.append(self.context.last_output)
            self.context.steps_taken.append(self.context.last_step)

        # Now check if we met our goal yet
        goal_message = {
            "Goal": self.context.latest_content,
            "Media Description": self.context.image_descriptions,
            "Plan": self.context.plan_dict,
            "Information Gathered": self.context.answer_output,
        }
        output = await self.agents.user_proxy.a_initiate_chat(
            recipient=self.agents.goal_judge,
            max_turns=1,
            message=f"(```{str(goal_message)}```",
        )

        if "##NOT YET##" not in output.chat_history[-1]["content"]:
            # We met our goal, so end the loop
            return None

        # We did not meet our goal, so obtain the next instruction
        message = {
            "Goal": self.context.latest_content,
            "Media Description": self.context.image_descriptions,
            "Plan": str(self.context.plan_dict),
            "Last Step": reflection_message,
            "Last Step Output": str(self.context.last_output),
            "Steps Taken": str(self.context.steps_taken),
        }
        output = await self.agents.user_proxy.a_initiate_chat(
            recipient=self.agents.reflection_assistant,
            max_turns=1,
            message=f"(```{str(message)}```",
        )
        return output.chat_history[-1]["content"]

    async def _execute_instruction(self, instruction):
        await self.eventer.emit_event(message="Executing step: " + instruction)
        prompt = instruction + (
            f"\n Contextual Information: \n{self.context.answer_output}"
            if self.context.answer_output
            else ""
        )
        output = await self.agents.user_proxy.a_initiate_chat(
            recipient=self.agents.assistant, max_turns=3, message=prompt
        )
        return [
            item["content"]
            for item in output.chat_history
            if item.get("name") == "Research_Assistant" and item["content"]
        ]

    async def _summarize_results(self):
        await self.eventer.emit_event(message="Summing up findings...")
        final_prompt = (
            f"Answer the user's query: {self.context.latest_content}\n\n"
            f"Images: {' '.join(self.context.image_descriptions)}\n"
            f"Use the following information only: {self.context.answer_output}"
        )
        final_output = await self.agents.user_proxy.a_initiate_chat(
            message=final_prompt, max_turns=1, recipient=self.agents.report_generator
        )
        return final_output.chat_history[-1]["content"]
