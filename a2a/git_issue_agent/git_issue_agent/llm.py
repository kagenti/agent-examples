from crewai import LLM
from git_issue_agent.config import Settings

class CrewLLM():
    def __init__(self, config: Settings):

        self.llm = LLM(
            model=config.TASK_MODEL_ID,
            base_url=config.LLM_API_BASE,
            api_key=config.LLM_API_KEY,
            **({'extra_headers': config.EXTRA_HEADERS} if config.EXTRA_HEADERS is not None and None not in config.EXTRA_HEADERS else {})
        )