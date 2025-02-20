from pydantic import BaseModel, Field
from typing import Dict, Union, List, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from company_researcher.nodes.cluster import Cluster
from company_researcher.utils.tavily_utils import TavilySearchInput, TavilyQuery

class InputState(BaseModel):
    company: str = Field(
        description="The name of the company to research",
        examples=["Tavily"],
    )
    company_url: str = Field(
        description="The official website URL of the company.",
        examples=["https://tavily.com/"],
    ),
    include: list[str] = Field(
        description=(
            "Optional list specifying information to include in the company research report, "
            "such as the company's official website URL, LinkedIn profile URL, headquarters location, "
            "number of employees, CEO's name, and more."
        ),
        examples=[
            "Company's official website URL",
            "Company's LinkedIn profile URL",
            "Location of headquarters formatted as <city>, <state code> (e.g. San Francisco, CA)",
            "Number of employees",
            "Name of the CEO"
        ],
        default_factory=list
    )


class OutputState(BaseModel):
    report: str = ""

class ResearchState(InputState, OutputState):
    grounding_data: Dict[str, Dict[str, Union[str, None]]] = Field(default_factory=dict)
    research_data: Dict[str, Dict[str, Union[str, float, None]]] = Field(default_factory=dict)
    clusters: List[Cluster] = Field(default_factory=list)
    chosen_cluster: int = Field(default_factory=int)
    search_queries: List[TavilyQuery] = Field(default_factory=list)
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)

