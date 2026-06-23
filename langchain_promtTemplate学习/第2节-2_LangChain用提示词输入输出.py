import os
from dotenv import load_dotenv , find_dotenv
_= load_dotenv(find_dotenv())
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(api_key=API_KEY,
    base_url=BASE_URL,
    timeout=30.0,  # 增加超时时间
    max_retries=2 )

prompt_template = """
我的名字叫【{name}】, 我的个人介绍是【{description}】,性别【{sex}】。
请根据我的名字和介绍，帮我想一段有吸引力的自我介绍的句子，以此来吸引读者关注和点赞我的账号。
"""

from langchain_core.prompts import PromptTemplate
template = PromptTemplate.from_template(prompt_template)
print("1.输出template的input_variables:")
print(template.input_variables)
prompt = template.format(name="同学小张",description = "热爱AI，持续学习",sex="男")
print("\n 2.输出format后完整的提示词:"+prompt)
response = llm.invoke(prompt)
print("3.llm调用提示词后的返回:"+response.content)



##1.2 ChatPromptTemplate: 创建一个Prompt的Message数组
print("##1.2 ChatPromptTemplate: 创建一个Prompt的Message数组")
print("核心区别\n"
"SystemMessagePromptTemplate：\n"
"设计用于系统消息，通常是给模型的指令、角色设定、系统约束等\n"
"对应 SystemMessage消息类型\n"
"HumanMessagePromptTemplate：\n"
"设计用于用户输入/人类消息\n"
"对应 HumanMessage消息类型 \n")
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate,HumanMessagePromptTemplate
template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("你是【{name}】的个人助手，你需要根据用户输入，"
   "来替用户生成一段有吸引力的自我介绍的句子，以此来吸引读者关注和点赞用户的账号。帮我写个rap"),
        HumanMessagePromptTemplate.from_template("{description}") ,
    ])
prompt = template.format(name = "同学狂龙",description ="请解释：这样的介绍的后果")
print("\n 提示词--template的变量的format赋值："+prompt)
response = llm.invoke(prompt)
print("\n 数组template的response:"+response.content)