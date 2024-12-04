import cohere
import asyncio

from company_researcher.nodes.cluster import Cluster


class RerankAgent:
    def __init__(self, cfg, utils):
        self.cfg = cfg
        self.utils = utils
        self.co = cohere.AsyncClient()

    async def rerank_documents(self, query, documents, top_n, timeout):
        """Performs reranking of documents using Cohere."""
        try:
            response = await asyncio.wait_for(
                self.co.rerank(
                    query=query,
                    documents=documents,
                    top_n=top_n,
                    return_documents=False,
                ),
                timeout=timeout,
            )
            return response.results
        except asyncio.TimeoutError:
            raise TimeoutError("Timeout occurred during reranking")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during reranking: {e}")

    def create_cluster(self, company_name, urls):
        """Creates a Cluster object from company name and URLs."""
        return Cluster(
            company_name=company_name,
            urls=urls,
        )

    async def run(self, state):
        """Main method to rerank research data and create clusters."""
        msg = "🔄 Reranking research data...\n"
        data = list(state.research_data.values())

        try:
            # Perform reranking
            query = f"Company {state.company}"
            documents = [result["content"] for result in data]
            top_n = self.cfg.DEFAULT_CLUSTER_SIZE
            timeout = self.cfg.RERANK_TIMEOUT

            rerank_results = await self.rerank_documents(
                query=query,
                documents=documents,
                top_n=top_n,
                timeout=timeout,
            )

            # Process results
            urls = []
            msg += "Top documents selected:\n"
            for r in rerank_results:
                original_result = data[r.index]
                msg += f"{original_result['url']}\n"
                urls.append(original_result["url"])

            # Create and return cluster
            cluster = self.create_cluster(state.company, urls)
            return {"clusters": [cluster], "messages": msg}

        except TimeoutError:
            return {"messages": "🚫 Timeout occurred while reranking research data"}
        except RuntimeError as e:
            return {"messages": f"🚫 Error during reranking: {e}"}
        except Exception as e:
            return {"messages": f"🚫 Unexpected error during reranking: {e}"}
