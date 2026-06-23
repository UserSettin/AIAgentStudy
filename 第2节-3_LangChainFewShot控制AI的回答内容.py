import os
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate

from dotenv import load_dotenv , find_dotenv
_= load_dotenv(find_dotenv())
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(api_key=API_KEY,
    base_url=BASE_URL,
    timeout=30.0,  # 增加超时时间
    max_retries=2 )
#例子（few-shot）
examples = [
    {
        "input" : "北京的天气怎么样",
        "output": "我控制了内容，只返回不怎样"
    },
{
        "input" : "美国强大吗",
        "output": "比中国强"
    },
]

#例子拼装的格式
example_prompt = PromptTemplate(input_variables=["input","output"],
                                template="输入:{input}\n输出:{output}")

#Prompt模板

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="输入:{input}\n输出 :",#Suffix是接受用户的输入，组装提问的prompt模版，就是在示例后添加的固定文本
    input_variables = ["input"] #input_variables表示用户输入的参数变量名

)

prompt = prompt.format(input = "羊城多少度")
print("===Prompt====")
print(prompt)
print("===llm调用prompt====")
response = llm.invoke(prompt)
print("===Response====")
print(response)


 #1. 准备示例数据
examples = [
    {
        "word": "美丽",
        "context": "形容风景",
        "synonym": "漂亮"
    },
    {
        "word": "聪明",
        "context": "形容小孩",
        "synonym": "机灵"
    },
    {
        "word": "巨大",
        "context": "形容建筑",
        "synonym": "庞大"
    }
]

# 2. 定义单个示例的格式
example_prompt = PromptTemplate(
    input_variables=["word", "context", "synonym"],
    template="词语: {word}\n语境: {context}\n同义词: {synonym}"
)

# 3. 创建FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="""请根据示例生成同义词：
词语: {word}
语境: {context}
同义词:""",
    input_variables=["word", "context"],
    example_separator="\n---\n"
)

# 4. 使用
formatted_prompt = few_shot_prompt.format(
    word="快速",
    context="形容动作"
)
print(formatted_prompt)
response = llm.invoke(formatted_prompt)

print(response)