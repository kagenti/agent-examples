from pydantic import BaseModel, Field
from typing import Literal, Optional

############
# Pydantic types for LLM response formats
############


class ChannelInfo(BaseModel):
    name: str = Field(description="The name of the slack channel")
    id: str = Field(description="The ID of the channel")
    description: str = Field(description="A description of the channel")


class ChannelList(BaseModel):
    channels: list[ChannelInfo]
    explanation: Optional[str] = Field(
        None, description="A detailed explanation as to why you chose this specific list of channels"
    )


class UserIntent(BaseModel):
    intent: Literal["LIST_CHANNELS", "QUERY CHANNELS"]


class UserRequirement(BaseModel):
    specific_channel_names: Optional[str] = Field(
        None, description="Specific channel names that the user would like to fetch"
    )
    types_of_channels: str = Field(
        description="A description of the types of channels that the user would like information about, if their request is not limited to specific channel names."
    )
    types_of_information_to_search: Optional[str] = Field(
        None,
        description="The types of information that the user wants to look for inside of the channels. \
                                                           Can be null if user is only interested in listing channels and not interested in searching inside channel content.",
    )
