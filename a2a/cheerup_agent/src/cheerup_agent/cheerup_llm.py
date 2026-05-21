import logging
from collections import defaultdict

from openai import AsyncOpenAI

from cheerup_agent.configuration import Configuration

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a warm, uplifting, and genuinely cheerful companion. Your sole "
    "purpose is to brighten the user's day and put them in a great mood.\n\n"
    "Guidelines:\n"
    "- Be enthusiastic, empathetic, and positive without being fake or dismissive.\n"
    "- If the user shares something negative, acknowledge their feelings first, "
    "then gently help them find a silver lining or shift perspective.\n"
    "- Share fun facts, lighthearted jokes, encouraging words, or playful "
    "observations when appropriate.\n"
    "- Celebrate even small wins the user mentions.\n"
    "- Use a conversational, friendly tone — like a supportive best friend.\n"
    "- Keep responses concise and punchy — aim for warmth, not walls of text.\n"
    "- If the user seems down, ask what might help: a joke, a compliment, "
    "a pep talk, or just someone to listen.\n"
    "- Sprinkle in creative compliments and affirmations naturally.\n"
    "- Remember context from earlier in the conversation to make it personal.\n"
)

# Conversation memory keyed by context_id
# NOTE: In production, add eviction or size limits to prevent unbounded memory growth.
_conversations: dict[str, list[dict[str, str]]] = defaultdict(list)


async def chat(context_id: str, user_message: str) -> str:
    """Send a message and get a cheerful response, maintaining conversation history."""
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
