import asyncio
from datetime import datetime
from tavily import AsyncTavilyClient
from typing import List, Optional
from pydantic import BaseModel, Field
from company_researcher.config import Config
cfg = Config()

# Define Tavily's arguments to tailor the search results
class TavilyQuery(BaseModel):
    query: str = Field(description="Web search query")
    search_depth: str = Field(description="The depth of the search, available options: 'basic', 'advanced'")
    time_range:  Optional[str] = Field(default=None, description="(Optional) Filters results by time range, useful when looking for sources like magazines and articles. Available options: 'day', 'week', 'month', 'year'.")
    include_domains: Optional[List[str]] = Field(default=None,
                                         description="List of domains to include in the research. Useful when trying to gather information from trusted and relevant domains")

class TavilySearchInput(BaseModel):
    sub_queries: List[TavilyQuery] = Field(description="Set of web search queries that can be answered in isolation")


class Tavily:
    def __init__(self):
        self.client = AsyncTavilyClient()

    async def extract(self, urls: list[str], sources_dict: dict, extract_depth="basic"):
        msg = ""

        async def process_batch(url_batch):
            batch_msg = ""
            try:
                response = await self.client.extract(urls=url_batch, extract_depth=extract_depth)
                for itm in response['results']:
                    url = itm['url']
                    raw_content = itm['raw_content']
                    if len(raw_content) > cfg.MAX_DOC_LENGTH:
                        raw_content = raw_content[:cfg.MAX_DOC_LENGTH] + " [...]"
                        if cfg.DEBUG:
                            print(f"Content from {url} was truncated to the maximum allowed length ({cfg.MAX_DOC_LENGTH} characters). Current length: {len(raw_content)}\nPreview:\n{raw_content}")
                    if url in sources_dict:
                        sources_dict[url]['raw_content'] = raw_content
                    else:
                        sources_dict[url] = {'raw_content': raw_content}
                    batch_msg += f"{url}\n"
                return batch_msg
            except Exception as e:
                return f"Error occurred during Tavily Extract request for batch: {e}\n"

        # Split URLs into batches of 20
        url_batches = [urls[i:i + 20] for i in range(0, len(urls), 20)]

        # Process all batches in parallel
        results = await asyncio.gather(*[process_batch(batch) for batch in url_batches])

        # Collect messages from all batches
        if results:
            msg += "Extracted raw content for:\n" + "".join(results)

        return sources_dict, msg

    async def search(self, sub_queries: List[TavilyQuery], sources_dict: dict):
        """
        Perform searches for each sub-query using the Tavily Search concurrently.

        :param sub_queries: List of search queries.
        :param sources_dict: Dictionary to store unique search results, keyed by URL.
        """

        # Define a coroutine function to perform a single search with error handling
        async def perform_search(query):
            try:
                print(query)
                # Add date to the query as we need the most recent results
                # query_with_date = f"{query.query} {datetime.now().strftime('%m-%Y')}"
                tavily_response = await self.client.search(query=query.query, topic="general", search_depth=query.search_depth, time_range=query.time_range, include_domains=query.include_domains, max_results=10)
                return tavily_response['results']
            except Exception as e:
                # Handle any exceptions, log them, and return an empty list
                if cfg.DEBUG:
                    print(f"Error occurred during search for query '{query}': {str(e)}")
                return []

        # Run all the search tasks in parallel
        search_tasks = [perform_search(itm) for itm in sub_queries]
        search_responses = await asyncio.gather(*search_tasks)

        # Combine the results from all the responses and update the sources_dict
        for response in search_responses:
            for result in response:
                url = result.get("url")
                if url and url not in sources_dict:
                    # Add the result to sources_dict if the URL is not already present
                    sources_dict[url] = result

        return sources_dict