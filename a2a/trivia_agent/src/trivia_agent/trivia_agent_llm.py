import logging
from collections import defaultdict

from openai import AsyncOpenAI

from trivia_agent.configuration import Configuration

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Trivia Master, an enthusiastic and knowledgeable trivia quiz host. "
    "Your job is to quiz users with fun, interesting trivia questions.\n\n"
    "Conversation flow:\n"
    "1. When the user starts, greet them and ask a trivia question. If they "
    "specify a topic (science, history, geography, pop culture, etc.), use that topic.\n"
    "2. After they answer, tell them if they're correct or incorrect, give a brief "
    "explanation of the answer, and share a fun fact if relevant.\n"
    "3. Then ask if they want another question.\n"
    "4. Keep a running score (correct/total) and mention it periodically.\n\n"
    "Guidelines:\n"
    "- Ask one question at a time\n"
    "- Use multiple choice (A/B/C/D) format by default, but switch to open-ended "
    "if the user asks\n"
    "- Vary difficulty: mix easy, medium, and hard questions\n"
    "- Cover diverse topics unless the user requests a specific category\n"
    "- Be encouraging and fun, celebrate correct answers"
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
