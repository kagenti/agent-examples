"""
Example of the business logic of an A2A agent for currency conversion.
"""

import logging
import os

from openai import AuthenticationError, InternalServerError

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError
from app.agent import CurrencyAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CurrencyAgentExecutor(AgentExecutor):
    """Currency Conversion AgentExecutor Example."""

    def __init__(self):
        self.agent = CurrencyAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        error = self._validate_request(context)
        if error:
            logger.warning(f"Invalid agent executor request: {context}")
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)
            logger.info(f"Created task for message : {context.message}")
            event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        try:
            async for item in self.agent.stream(query, task.context_id):
                is_task_complete = item["is_task_complete"]
                require_user_input = item["require_user_input"]

                if not is_task_complete and not require_user_input:
                    logger.info(f"Updating status for non-input task: {task.id}")
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            item["content"],
                            task.context_id,
                            task.id,
                        ),
                    )
                elif require_user_input:
                    logger.info(f"Updating status for input task: {task.id}")
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            item["content"],
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    break
                else:
                    logger.info("Adding artifact for item")
                    await updater.add_artifact(
                        [Part(root=TextPart(text=item["content"]))],
                        name="conversion_result",
                    )
                    await updater.complete()
                    break

        except InternalServerError as e:
            msg = f"""CurrencyAgentExecutor reports an InternalServerError error.

This can happen if the agent's LLM_API_BASE environment variable does not point to an OpenAI server.

LLM_API_BASE is {os.getenv("LLM_API_BASE", "undefined")}

Use `kubectl -n <namespace> logs deployment/<agent-name>` for details.
"""
            logger.error(msg=msg)
            logger.error(msg=f"Raw InternalServerError: {e}")
            await updater.update_status(
                TaskState.input_required,
                new_agent_text_message(
                    msg,
                    task.context_id,
                    task.id,
                ),
                final=True,
            )

        except AuthenticationError as e:
            msg = f"""CurrencyAgentExecutor reports an authentication error.

When importing this agent into Kagenti, expand Environment Variables and Add Variable,
or import https://github.com/kagenti/agent-examples/blob/main/a2a/a2a_currency_converter/.env.openai

Use `kubectl -n <namespace> logs deployment/<agent-name>` for details.

Also check
`kubectl -n <namespace> get secret openai-secret -o jsonpath="{"{"}.data.apikey{"}"}" | base64 -d`
The key should match your OpenAI key."""
            logger.error(msg=msg)
            logger.error(msg=f"Raw AuthenticationError {e}")
            await updater.update_status(
                TaskState.input_required,
                new_agent_text_message(
                    msg,
                    task.context_id,
                    task.id,
                ),
                final=True,
            )

        except Exception as e:
            logger.error(f"An error occurred while streaming the response: {e}")
            logger.info(msg=f"The error is a {type(e)}")
            await updater.update_status(
                TaskState.input_required,
                new_agent_text_message(
                    # We don't show the error to the user, as it may have credentials
                    """Internal error on the agent.
                    Use `kubectl -n <namespace> logs deployment/<agent-name>` for details""",
                    task.context_id,
                    task.id,
                ),
                final=True,
            )
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, _: RequestContext) -> bool:
        return False

    async def cancel(self, _: RequestContext, event_queue: EventQueue) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
