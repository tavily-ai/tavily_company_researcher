from langgraph.graph import StateGraph, END

from company_researcher.config import Config
from company_researcher.state import InputState, OutputState, ResearchState
from company_researcher.nodes import GroundAgent, ResearchAgent, ClusterAgent, EnrichAgent, WriteAgent
from company_researcher.utils.all import Utils

cfg = Config()
utils = Utils()

# Initialize agents
ground_agent = GroundAgent(cfg, utils)
research_agent = ResearchAgent(cfg, utils)
cluster_agent = ClusterAgent(cfg, utils)
enrich_agent = EnrichAgent(cfg, utils)
write_agent = WriteAgent(cfg, utils)

# Define a Langchain graph
workflow = StateGraph(ResearchState, input=InputState, output=OutputState)

# Add node for each agent
workflow.add_node('ground', ground_agent.run)
workflow.add_node('research', research_agent.run)
workflow.add_node('cluster', cluster_agent.run)
workflow.add_node('enrich', enrich_agent.run)
workflow.add_node('write', write_agent.run)

# Set up edges
workflow.add_edge('ground', 'research')
workflow.add_edge('research', 'cluster')
workflow.add_edge('cluster', 'enrich')
workflow.add_edge('enrich', 'write')
workflow.add_edge('write', END)

# Set up start node
workflow.set_entry_point('ground')

graph = workflow.compile()
graph.name = "Tavily Company Researcher"
