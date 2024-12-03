from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, HumanMessage, ToolMessage
import time

class Cluster(BaseModel):
    company_name: str = Field(
        ...,
        description="The name or identifier of the company these documents belong to."
    )
    urls: List[str] = Field(
        ...,
        description="A list of URLs relevant to the identified company."
    )

class Clusters(BaseModel):
    clusters: List[Cluster] = Field(default_factory=list, description="List of clusters")

class ClusterAgent:
    def __init__(self, cfg, utils):
        self.cfg = cfg
        self.utils = utils

    async def cluster(self, company, company_url, grounding_data, research_data):
        target_domain = company_url.split("//")[-1].split("/")[0]
        prompt = f"""
                We conducted a search for a company called '{company}', but the results may include documents from other companies with similar names or domains.
                Your task is to accurately categorize these retrieved documents based on which specific company they pertain to, using the initial company information as "ground truth."

                ### Target Company Information
                - **Company Name**: '{company}'
                - **Primary Domain**: '{target_domain}'
                - **Initial Context (Ground Truth)**: Information below should act as a verification baseline. Use it to confirm that the document content aligns directly with {company}.
                - **{grounding_data}**

                ### Retrieved Documents for Clustering
                Below are the retrieved documents, including URLs and brief content snippets:
                {[{'url': doc['url'], 'snippet': doc['content']} for doc in research_data.values()]}

                ### Clustering Instructions
                - **Primary Domain Priority**: Documents with URLs containing '{target_domain}' should be prioritized for the main cluster for '{company}'.
                - **Include Relevant Third-Party Sources**: Documents from third-party domains (e.g., news sites, industry reports) should also be included in the '{company}' cluster if they provide specific information about '{company}', reference '{target_domain}', or closely match the initial company context.
                - **Separate Similar But Distinct Domains**: Documents from similar but distinct domains (e.g., '{target_domain.replace('.com', '.io')}') should be placed in separate clusters unless they explicitly reference the target domain and align with the company's context.
                - **Handle Ambiguities Separately**: Documents that lack clear alignment with '{company}' should be placed in an "Ambiguous" cluster for further review.

                ### Example Output Format
                {{
                    "clusters": [
                        {{
                            "company_name": "Name of Company A",
                            "urls": [
                                "http://example.com/doc1",
                                "http://example.com/doc2"
                            ]
                        }},
                        {{
                            "company_name": "Name of Company B",
                            "urls": [
                                "http://example.com/doc3"
                            ]
                        }},
                        {{
                            "company_name": "Ambiguous",
                            "urls": [
                                "http://example.com/doc4"
                            ]
                        }}
                    ]
                }}

                ### Key Points
                - **Focus on Relevant Content**: Documents that contain relevant references to '{company}' (even from third-party domains) should be clustered with '{company}' if they align well with the initial information and context provided.
                - **Identify Ambiguities**: Any documents without clear relevance to '{company}' should be placed in the "Ambiguous" cluster for manual review.
            """
        try:
            messages = [SystemMessage(content=prompt)]
            response = await self.cfg.model.with_structured_output(Clusters).ainvoke(messages)
            clusters = response.clusters  # Access the structured clusters directly
            return clusters, ""
        except Exception as e:
            msg = f"🚫 Error clustering: {str(e)}\n"
            clusters = []
            return clusters, msg

    # Define the function to automatically choose the correct cluster, can add in the future manual selection support
    async def choose_cluster(self, company_url, clusters):
        chosen_cluster = 0
        for index, cluster in enumerate(clusters):
            # Check if any URL in the cluster starts with the company URL
            if any(company_url in url for url in cluster.urls):
                chosen_cluster = index
                break

        cluster = clusters[chosen_cluster]
        msg = f"Automatically selected cluster for '{company_url}' as {cluster.company_name}.\n"
        return chosen_cluster, msg


    async def run(self, state):
        msg = "📊 Beginning clustering process...\n"
        start = time.time()
        clusters, cluster_msg = await self.cluster(state.company, state.company_url, state.grounding_data, state.research_data)
        end = time.time()
        print("clustering response time: ", end-start)
        chosen_cluster, choose_msg = await self.choose_cluster(state.company_url, clusters)
        return {"clusters": clusters, "chosen_cluster": chosen_cluster, "messages": msg + cluster_msg + choose_msg}
