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
        self.MAX_DOC_LENGTH = 8000
        self.BASE_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=2000)
        self.FACTUAL_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=2000)
        self.DEBUG = False