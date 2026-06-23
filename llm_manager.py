# llm_manager.py
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

_ = load_dotenv(find_dotenv())

class LLMManager:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, timeout=30.0, max_retries=2):
        # initialize the ChatOpenAI llm once
        if not hasattr(self, 'llm'):
            self.llm = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                timeout=timeout,
                max_retries=max_retries
            )