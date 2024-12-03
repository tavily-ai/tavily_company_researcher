from pydantic import BaseModel, Field
from typing import Dict, Union, List, Annotated, Literal
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from src.nodes.cluster import Cluster

class InputState(BaseModel):
    company: str = Field(
        description="The name of the company to research",
        examples=["Tavily"],
    )
    company_url: str = Field(
        description="The official website URL of the company.",
        examples=["https://tavily.com/"],
    )

class OutputState(BaseModel):
    report: str = ""

class ResearchState(InputState, OutputState):
    grounding_data: Dict[str, Dict[str, Union[str, None]]] = Field(default_factory=dict)
    research_data: Dict[str, Dict[str, Union[str, float, None]]] = Field(default_factory=dict)
    clusters: List[Cluster] = Field(default_factory=list)
    chosen_cluster: int = Field(default_factory=int)
    search_queries: List[str] = Field(default_factory=list)
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)

