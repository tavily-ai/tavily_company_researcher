from langchain_openai import ChatOpenAI

# Description: Configuration file
class Config:
    def __init__(self):
        """
        Initializes the configuration for the agent
        """
        self.MAX_SEARCH_QUERIES = 3
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1000)