from langchain_openai import ChatOpenAI

# Description: Configuration file
class Config:
    def __init__(self):
        """
        Initializes the configuration for the agent
        """
        self.MAX_SEARCH_QUERIES = 6
        self.DEFAULT_CLUSTER_SIZE = 10
        self.RERANK_TIMEOUT = 3
        self.MAX_PROMPT_LENGTH = 350000
        self.MAX_GROUND_LENGTH = 4000
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1000)
        self.DEBUG = False