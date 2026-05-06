from pydantic_settings import BaseSettings


class Configuration(BaseSettings):
    llm_model: str = "llama3.1"
    llm_api_base: str = "http://localhost:11434/v1"
    llm_api_key: str = "dummy"
    workspace_root: str = "/workspace"
    checkpoint_db_url: str = "memory"
    context_ttl_days: int = 7

    # Per-node model overrides (empty = use llm_model default)
    llm_model_planner: str = ""
    llm_model_executor: str = ""
    llm_model_reflector: str = ""
    llm_model_reporter: str = ""
    llm_model_thinking: str = ""  # bare LLM for thinking iterations
    llm_model_micro_reasoning: str = ""  # LLM+tools for micro-reasoning

    def model_for_node(self, node: str) -> str:
        """Return the model to use for a specific node type."""
        overrides = {
            "planner": self.llm_model_planner,
            "executor": self.llm_model_executor,
            "reflector": self.llm_model_reflector,
            "reporter": self.llm_model_reporter,
            "thinking": self.llm_model_thinking,
            "micro_reasoning": self.llm_model_micro_reasoning,
        }
        return overrides.get(node, "") or self.llm_model
