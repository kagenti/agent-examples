from pydantic import BaseModel, Field
from typing import Literal, Optional

############
# Pydantic types for LLM response formats
############


class RepositoryJudgement(BaseModel):
    is_owner_and_repo_identified: bool = Field(False, description="Whether the user's query indicates both an owner/organization name and the repo name")
    explanation: str = Field(None, description="A detailed explanation as to why you made the judgement")