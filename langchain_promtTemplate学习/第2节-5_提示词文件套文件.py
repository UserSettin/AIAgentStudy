#LangChain也允许你在promt文件中再套Prompt文件：
#将文件中的template字段单独放到一个txt文件。拆分后文件如下：
#prompt_template_test.txt

import json
import os
from llm_manager import LLMManager
from langchain_core.prompts import PromptTemplate

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载配置文件并构建 PromptTemplate
def load_prompt_from_config(path):
    # 如果path不是绝对路径，则转换为脚本所在目录的绝对路径
    if not os.path.isabs(path):
        path = os.path.join(SCRIPT_DIR, path)
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    # 处理template_path时也需要转换为绝对路径
    template_path = config.get("template_path")
    if template_path and not os.path.isabs(template_path):
        template_path = os.path.join(SCRIPT_DIR, template_path)
    template = config.get("template") or open(template_path, encoding="utf-8").read()
    return PromptTemplate.from_template(template)

prompt = load_prompt_from_config("第2节-5_langchain_prompt_file_test.json")

prompt_str = prompt.format(name="狂龙老总", description="热爱吹牛逼")
print(prompt_str)

llm = LLMManager().llm
response = llm.invoke(prompt_str)
print(f"\n{response}")