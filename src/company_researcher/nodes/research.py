from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, HumanMessage, ToolMessage


class TavilySearchInput(BaseModel):
    sub_queries: List[str] = Field(description="set of web search queries that can be answered in isolation")


class ResearchAgent:
    def __init__(self, cfg, utils):
        self.cfg = cfg
        self.utils = utils

    async def generate_queries(self, state):
        try:
            msg = f"🤔 Generating search queries based on grounding data...\n"
            if self.cfg.DEBUG:
                print(msg)
            prompt = (
                f"You are an expert company researcher specializing in generating company analysis reports.\n"
                f"Your task is to generate up to {self.cfg.MAX_SEARCH_QUERIES} precise **web search queries** to thoroughly understand the company: '{state.company}'.\n\n"
                f"### Key Areas to Explore:\n"
                f"- **Company Background**: Focus on keywords such as history, mission, headquarters, CEO, leadership team, and number of employees.\n"
                f"- **Products and Services**: Search for offerings like main products, unique features, customer segments, and market differentiation.\n"
                f"- **Market Position**: Use terms like market competition, industry ranking, competitive landscape, market reach, and impact.\n"
                f"- **Financials**: Look for information on funding rounds, revenue, financial growth, recent investments, and performance metrics.\n\n"
            )

            if state.include:
                prompt += (
                    f"### Required Information to Include:\n"
                    f"- You are tasked with ensuring the following specific types of information are covered in the report, as specified by the user:\n"
                    f"{', '.join(state.include)}\n"
                    # f"- Prioritize missing information: Check the grounding data and identify any missing elements from the required information to include.\n"
                    f"- Generate a search query only for the information that is missing from the provided grounding data.\n"
                )

            prompt += (
                f"### Grounding Data:\n"
                f"Use the grounding data provided from the company's website below to ensure queries are closely tied to **{state.company}** and reflect its latest context:\n"
                f"{state.grounding_data}\n\n"
                f"### Additional Guidance:\n"
            )

            # if state.include:
            #     prompt += (
            #         f"- Prioritize missing information: Check the grounding data and identify any missing elements from the required information to include.\n"
            #     )

            prompt += (
                f"- Ensure each query incorporates **specific keywords** derived from the grounding data, such as the company's name, key products or services, leadership titles, geographical locations, and other unique identifiers, to allow the search engine to retrieve the most relevant sources specific to the company you are researching.\n"
                f"- **Limit each query to 100 characters or fewer** to ensure clarity and search engine compatibility.\n"
                f"- Structure queries to focus on specific aspects of the company, such as \"{state.company} number of employees\" or \"{state.company} market competition.\"\n"
                f"- Avoid redundancy: Each query should focus on unique information to retrieve relevant details efficiently. For example, there should be only one query to search for the company's CEO name.\n"

            )
            if self.cfg.DEBUG:
                print(prompt)
            messages = [SystemMessage(content=prompt)]
            response = await self.cfg.BASE_LLM.with_structured_output(TavilySearchInput).ainvoke(messages)
            return response.sub_queries, msg
        except Exception as e:
            msg = f"🚫 An error occurred during search queries generation: {str(e)}"
            return [f"Company {state.company}"], msg

    async def run(self, state):
        sub_queries, msg = await self.generate_queries(state)
        msg += "🔎 Tavily Searching ...\n" + "\n".join(f'"{query}"' for query in sub_queries)
        if self.cfg.DEBUG:
            print(msg)
        research_data = await self.utils.tavily.search(sub_queries, state.research_data)
        return {"messages": msg, "search_queries": sub_queries, "research_data": research_data}
