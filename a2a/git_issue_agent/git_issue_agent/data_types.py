import json

from pydantic import BaseModel, Field, field_validator

############
# Pydantic types for LLM response formats
############


class IssueSearchInfo(BaseModel):
    owner: str | None = Field(None, description="The issue owner or organization.")
    repo: str | None = Field(None, description="The specified repository. Leave blank if none specified.")
    issue_numbers: list[int] | None = Field(
        None, description="Specific issue number(s) mentioned by the user. If none mentioned leave blank."
    )

    @field_validator("issue_numbers", mode="before")
    @classmethod
    def coerce_string_to_list(cls, v):
        """Small LLMs often serialize arrays as strings in tool call args."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        return v
