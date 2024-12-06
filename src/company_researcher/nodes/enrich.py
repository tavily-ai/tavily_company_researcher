class EnrichAgent:
    def __init__(self, cfg, utils):
        self.cfg = cfg
        self.utils = utils

    async def run(self, state):
        clusters = state.clusters
        chosen_cluster = clusters[state.chosen_cluster]
        msg = f"🚀 Enriching documents for selected cluster '{chosen_cluster.company_name}'...\n"
        if self.cfg.DEBUG:
            print(msg)
        research_data, extract_msg = await self.utils.tavily.extract(chosen_cluster.urls, state.research_data)
        if self.cfg.DEBUG:
            print(extract_msg)
        return {"research_data": research_data, "messages": msg + extract_msg}

