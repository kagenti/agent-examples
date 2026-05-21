from slack_researcher.config import Settings
from slack_researcher.data_types import ChannelList, UserIntent, UserRequirement


class LLMConfig:
    def __init__(self, config: Settings):
        self._base_config = {
            "model": config.TASK_MODEL_ID,
            "base_url": config.LLM_API_BASE,
            "api_type": "openai",
            "api_key": config.LLM_API_KEY,
        }

        self.openai_llm_config = self._create_llm_config(config, None)
        self.channel_llm_config = self._create_llm_config(config, ChannelList)
        self.intent_classifier_llm_config = self._create_llm_config(config, UserIntent)
        self.user_requirement_llm_config = self._create_llm_config(config, UserRequirement)

    def _create_llm_config(self, config, response_format):
        return {
            "config_list": [
                {
                    **self._base_config,
                    **({"response_format": response_format} if response_format else {}),
                    **({"default_headers": config.EXTRA_HEADERS} if config.EXTRA_HEADERS else {}),
                }
            ],
            "temperature": config.MODEL_TEMPERATURE,
        }
