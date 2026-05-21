import logging
from collections import defaultdict

from openai import AsyncOpenAI

from recipe_agent.configuration import Configuration

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a friendly recipe assistant. Your job is to help users figure out "
    "what to cook for dinner based on the ingredients they have.\n\n"
    "Conversation flow:\n"
    "1. First, ask the user what ingredients they have in their fridge.\n"
    "2. Once they list ingredients, ask about any dietary preferences or "
    "restrictions (vegetarian, allergies, cuisine preference, etc.).\n"
    "3. Then suggest 2-3 dinner recipes using those ingredients, with brief "
    "instructions.\n"
    "4. If the user picks a recipe, give detailed step-by-step instructions.\n\n"
    "Be concise but helpful. If the user already provided ingredients in their "
    "first message, skip step 1 and move to step 2 or 3."
)

# Conversation memory keyed by context_id
# NOTE: In production, add eviction or size limits to prevent unbounded memory growth.
_conversations: dict[str, list[dict[str, str]]] = defaultdict(list)


async def chat(context_id: str, user_message: str) -> str:
    """Send a message and get a response, maintaining conversation history."""
    config = Configuration()

    client = AsyncOpenAI(
        base_url=config.llm_api_base,
        api_key=config.llm_api_key,
    )

    history = _conversations[context_id]
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    logger.info("Sending %d messages to LLM for context %s", len(messages), context_id)

    response = await client.chat.completions.create(
        model=config.llm_model,
        messages=messages,
    )

    assistant_message = response.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_message})

    logger.info("LLM response for context %s: %s", context_id, assistant_message[:200])
    return assistant_message
