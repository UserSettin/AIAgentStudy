import os
from dotenv import load_dotenv, find_dotenv
_=load_dotenv(find_dotenv())
from langchain_openai import ChatOpenAI
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

#这里开始是调用langchain来调OepnAi对话
llm = ChatOpenAI(api_key=API_KEY,
    base_url=BASE_URL,
    timeout=30.0,  # 增加超时时间
    max_retries=2 )
response = llm.invoke("你是谁" )#通过invoke传入对话
print("普通调用："+response.content)

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
message = [
    SystemMessage(content ="你是[同学小张]的个人助理，你叫小明") ,
    HumanMessage(content = "我叫狂龙哥"),
    HumanMessage(content="我是谁"),
    AIMessage(content = "狂龙哥，有什么吩咐"),


]
response = llm.invoke(message)
print(response.content)