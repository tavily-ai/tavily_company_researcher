from typing import Literal

def cluster_router(state) -> Literal["enrich", "rerank"]:
    """Routes the workflow after the 'cluster' step.

     If no clusters are formed, it falls back to 'rerank' for reevaluation."""
    if state.clusters:
        return "enrich"
    else:
        return "rerank"

def rerank_router(state) -> Literal["enrich", "write"]:
    """Routes the workflow after the 'rerank' step.

    If no clusters are formed even after reranking, it skips to the 'write' step without enriching the documents."""
    if state.clusters:
        return "enrich"
    else:
        return "write"
