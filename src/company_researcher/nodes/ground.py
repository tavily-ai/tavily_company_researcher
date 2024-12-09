class GroundAgent:
    def __init__(self, cfg, utils):
        self.cfg = cfg
        self.utils = utils

    async def run(self, state):
        msg = f"🔗 Initiating initial grounding for company '{state.company}'...\n"
        if self.cfg.DEBUG:
            print(msg)
        grounding_data, extract_msg = await self.utils.tavily.extract([state.company_url], state.grounding_data, use_cache=False)
        if self.cfg.DEBUG:
            print(extract_msg)
        if not grounding_data:
            grounding_data, extract_msg = await self.utils.tavily.extract([state.company_url], state.grounding_data,
                                                                          extract_depth="advanced")
            if self.cfg.DEBUG:
                print("Used advanced grounding")
        return {"grounding_data": grounding_data, "messages": msg + extract_msg}



