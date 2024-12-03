from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, HumanMessage, ToolMessage


class TavilySearchInput(BaseModel):
    sub_queries: List[str] = Field(description="set of sub-queries that can be answered in isolation")


class ResearchAgent:
    def __init__(self, cfg, utils):
        self.cfg = cfg
        self.utils = utils

    async def generate_queries(self, state):
        try:
            msg = "🤔 Generating search queries based on grounding data...\n"
            prompt = f"""You are an expert company researcher specializing in generating company analysis reports.
            Your task is to generate {self.cfg.MAX_SEARCH_QUERIES} sub-queries to thoroughly understand the company: '{state.company}'.

            ### Key Areas to Explore:
            - **Company Background**: Focus on keywords such as history, mission, headquarters, CEO, leadership team, and number of employees.
            - **Products and Services**: Search for offerings like main products, unique features, customer segments, and market differentiation.
            - **Market Position**: Use terms like market competition, industry ranking, competitive landscape, market reach, and impact.
            - **Financials**: Look for information on funding rounds, revenue, financial growth, recent investments, and performance metrics.

            Use the grounding data provided from the company's website below to ensure queries are closely tied to **{state.company}** and reflect its latest context:
            {state.grounding_data}
            
            ### Additional Guidance:
            - Ensure each query incorporates **specific keywords** derived from the grounding data, such as the company's name, key product or service names, leadership titles, geographical locations, and other unique identifiers.
            - These keywords should directly associate with the company to enhance the search accuracy and relevance.
            - Each query must be clear, concise, and designed to find the most relevant sources for this specific company and its context.

            """
            #      Make sure each query is targeted, using inferred keywords from the grounding data (e.g., company name, key product/service names, locations, or leadership titles) to find the most relevant sources for this specific company.
            #
            #             Ensure that the queries are clear, concise, and directly aligned with the company's context.
            messages = [SystemMessage(content=prompt)]
            response = await self.cfg.model.with_structured_output(TavilySearchInput).ainvoke(messages)
            return response.sub_queries, msg
        except Exception as e:
            msg = f"🚫 An error occurred during search queries generation: {str(e)}"
            return [f"Company {state.company}"], msg

    async def run(self, state):
        sub_queries, msg = await self.generate_queries(state)
        msg += "🔎 Tavily Searching ...\n" + "\n".join(f'"{query}"' for query in sub_queries)
        research_data = await self.utils.tavily.search(sub_queries, state.research_data)
        return {"messages": msg, "search_queries": sub_queries, "research_data": research_data}
