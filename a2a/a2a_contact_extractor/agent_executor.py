import logging

from a2a.helpers import (
    new_data_artifact,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractorAgentExecutor(AgentExecutor):
    """
    A ExtractorAgent agent executor.
    """

    def __init__(self, agent):
        self.agent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)

        async for item in self.agent.stream(query, task.context_id):
            is_task_complete = item["is_task_complete"]
            require_user_input = item["require_user_input"]
            # content = item["content"]
            content = item.get("content", "")

            logger.info(
                f"Stream item received: complete={is_task_complete}, require_input={require_user_input}, content_len={len(content)}"
            )

            agent_outcome = await self.agent.invoke(query, task.context_id)
            is_task_complete = agent_outcome["is_task_complete"]
            require_user_input = not is_task_complete
            content = agent_outcome.get("text_parts", [])
            data = agent_outcome.get("data", {})
            # Ensure content is a string by extracting text from TextPart objects
            if isinstance(content, list):
                # Extract the text from each TextPart object
                content = " ".join(part.text for part in content)

            if data:
                parts = new_data_artifact(
                    name="current_result",
                    description="Result of request to agent.",
                    data=data,
                ).parts
            else:
                parts = [new_text_part(content)]

            if require_user_input:
                await task_updater.update_status(
                    TaskState.TASK_STATE_INPUT_REQUIRED,
                    new_text_message(
                        content,
                        context_id=task.context_id,
                        task_id=task.id,
                    ),
                )
            elif is_task_complete:
                await task_updater.add_artifact(
                    parts,
                    name="current_result",
                    last_chunk=True,
                )
                await task_updater.complete()
            else:
                await task_updater.update_status(
                    TaskState.TASK_STATE_WORKING,
                    new_text_message(
                        "Analyzing your text...",
                        context_id=task.context_id,
                        task_id=task.id,
                    ),
                )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")
