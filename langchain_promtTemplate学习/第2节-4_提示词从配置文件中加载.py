import json
import os

from llm_manager import LLMManager
from langchain_core.prompts import PromptTemplate

# 获取脚本所在目录的绝对路径，确保无论从哪里运行都能找到配置文件
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "第2节-4提示词.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    prompt_config = json.load(f)

# Build a PromptTemplate from the dict
if isinstance(prompt_config, dict):
    template_text = prompt_config.get("template")
    input_vars = prompt_config.get("input_variables", [])
    if not template_text:
        raise RuntimeError("Prompt config does not contain a 'template' field.")
    prompt = PromptTemplate(input_variables=input_vars, template=template_text)
else:
    # If the JSON file contained a path string or other data, raise a helpful error
    raise RuntimeError("Expected prompt config to be a dict with 'template' and 'input_variables'.")

prompt_str = prompt.format(name="同学小张", description="热爱AI,持续学习，持续干货输出")
print(prompt_str)

llm = LLMManager().llm
response = llm.invoke(prompt_str)
print(f"\n{response}")