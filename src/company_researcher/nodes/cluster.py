import json
from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, HumanMessage, ToolMessage

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

    async def cluster(self, state):
        target_domain = state.company_url.split("//")[-1].split("/")[0]

        prompt = (
            f"We conducted a search for a company called '{state.company}', but the results may include documents from other companies with similar names or domains.\n"
            f"Your task is to accurately categorize these retrieved documents based on which specific company they pertain to, using the initial company information as 'ground truth.'\n\n"
            f"### Target Company Information\n"
            f"- **Company Name**: '{state.company}'\n"
            f"- **Primary Domain**: '{target_domain}'\n"
            f"- **Initial Context (Ground Truth)**: Information below should act as a verification baseline. Use it to confirm that the document content aligns directly with {state.company}.\n"
            f"- **{json.dumps(state.grounding_data)}**\n\n"
            f"### Retrieved Documents for Clustering\n"
            f"Below are the retrieved documents, including URLs and brief content snippets:\n"
            f"{[{'url': doc['url'], 'snippet': doc['content']} for doc in state.research_data.values()]}\n\n"
            f"### Clustering Instructions\n"
            f"- **Primary Domain Priority**: Documents with URLs containing '{target_domain}' should be prioritized for the main cluster for '{state.company}'.\n"
            f"- **Include Relevant Third-Party Sources**: Documents from third-party domains (e.g., news sites, industry reports) should also be included in the '{state.company}' cluster if they provide specific information about '{state.company}', reference '{target_domain}', or closely match the initial company context.\n"
        )

        if state.include:
            prompt += (
                f"- **Trusted Sources Inclusion**: If possible, trusted sources that include the following information should be added to the main cluster:\n"
                f"{', '.join(state.include)}.\n"
            )

        prompt += (
            f"- **Separate Similar But Distinct Domains**: Documents from similar but distinct domains (e.g., '{target_domain.replace('.com', '.io')}') should be placed in separate clusters unless they explicitly reference the target domain and align with the company's context.\n"
            f"- **Handle Ambiguities Separately**: Documents that lack clear alignment with '{state.company}' should be placed in an 'Ambiguous' cluster for further review.\n\n"
            f"### Example Output Format\n"
            f"{{\n"
            f"    'clusters': [\n"
            f"        {{\n"
            f"            'company_name': 'Name of Company A',\n"
            f"            'urls': [\n"
            f"                'http://example.com/doc1',\n"
            f"                'http://example.com/doc2'\n"
            f"            ]\n"
            f"        }},\n"
            f"        {{\n"
            f"            'company_name': 'Name of Company B',\n"
            f"            'urls': [\n"
            f"                'http://example.com/doc3'\n"
            f"            ]\n"
            f"        }},\n"
            f"        {{\n"
            f"            'company_name': 'Ambiguous',\n"
            f"            'urls': [\n"
            f"                'http://example.com/doc4'\n"
            f"            ]\n"
            f"        }}\n"
            f"    ]\n"
            f"}}\n\n"
            f"### Key Points\n"
            f"- **Focus on Relevant Content**: Documents that contain relevant references to '{state.company}' (even from third-party domains) should be clustered with '{state.company}' if they align well with the initial information and context provided.\n"
            f"- **Identify Ambiguities**: Any documents without clear relevance to '{state.company}' should be placed in the 'Ambiguous' cluster for manual review.\n"
        )
        prompt = prompt[:self.cfg.MAX_PROMPT_LENGTH]
        if self.cfg.DEBUG:
            print(prompt)
        try:
            messages = [SystemMessage(content=prompt)]
            response = await self.cfg.BASE_LLM.with_structured_output(Clusters).ainvoke(messages)
            clusters = response.clusters  # Access the structured clusters directly
            return clusters, ""
        except Exception as e:
            msg = f"🚫 Error accrued during clustering: {str(e)}\n"
            clusters = []
            return clusters, msg

    # Define the function to automatically choose the correct cluster, can add in the future manual selection support
    async def choose_cluster(self, company_url, clusters):
        chosen_cluster = 0
        msg = ""
        for index, cluster in enumerate(clusters):
            # Check if any URL in the cluster starts with the company URL
            if any(company_url in url for url in cluster.urls):
                chosen_cluster = index
                break
        if clusters:
            cluster = clusters[chosen_cluster]
            msg = f"Automatically selected cluster: {cluster.company_name} with the following urls: {cluster.urls}\n"
        return chosen_cluster, msg


    async def run(self, state):
        msg = "📊 Beginning clustering process...\n"
        if self.cfg.DEBUG:
            print(msg)
        clusters, cluster_msg = await self.cluster(state)
        if self.cfg.DEBUG:
            print(cluster_msg)
        chosen_cluster, choose_msg = await self.choose_cluster(state.company_url, clusters)
        if self.cfg.DEBUG:
            print(choose_msg)
        return {"clusters": clusters, "chosen_cluster": chosen_cluster, "messages": msg + cluster_msg + choose_msg}
