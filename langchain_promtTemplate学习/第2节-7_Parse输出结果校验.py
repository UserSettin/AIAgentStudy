# LangChain Parse部分：输出结果校验的封装
# 本文件演示如何使用OutputParser校验和格式化大模型的输出

# 1. 加载环境变量和导入模块
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# 导入LLM相关模块
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 导入输出解析器相关模块
# StrOutputParser: 返回原始字符串输出
# PydanticOutputParser: 解析为Pydantic模型，支持复杂结构和类型验证
# OutputFixingParser: 自动修复格式错误的输出（在langchain_classic包中）
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser

# 导入Pydantic模型定义模块
from pydantic import BaseModel, Field, ValidationError
from typing import List

# 2. 初始化大模型实例
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=30.0,
    max_retries=2
)

# ==============================================
# 3.1 使用步骤
# ==============================================
# 输出解析器的基本使用流程：
# 1. 定义期望的输出结构（如Pydantic模型）
# 2. 创建对应的解析器
# 3. 将解析器添加到Chain中
# 4. 获取结构化的输出结果

print("=" * 60)
print("3.1 使用步骤 - 输出解析器基础用法")
print("=" * 60)

# 示例：使用StrOutputParser获取字符串输出
print("\n示例1 - StrOutputParser：获取原始字符串")
print("-" * 40)
print("""
StrOutputParser是最简单的解析器，它直接返回模型输出的字符串内容。
在没有解析器时，llm.invoke()返回的是一个AIMessage对象，
包含content、response_metadata等字段。
StrOutputParser可以帮我们直接提取content部分。
""")

# 创建StrOutputParser
str_parser = StrOutputParser()

# 构建Chain: prompt → llm → parser
# 使用LCEL语法（LangChain Expression Language）
prompt = [
    SystemMessage(content="请用一句话回答"),
    HumanMessage(content="什么是人工智能？")
]

# 方式1：不使用解析器
response_no_parser = llm.invoke(prompt)
print(f"不使用解析器的返回类型: {type(response_no_parser).__name__}")
print(f"不使用解析器的返回值: {response_no_parser}")

# 方式2：使用解析器
response_with_parser = str_parser.invoke(response_no_parser)
print(f"\n使用解析器的返回类型: {type(response_with_parser).__name__}")
print(f"使用解析器的返回值: {response_with_parser}")

# ==============================================
# 3.2 不符合要求的情况
# ==============================================
# 当大模型输出不符合预期格式时，解析器会抛出错误。
# 常见问题：
# 1. JSON格式错误（如缺少引号、逗号）
# 2. 字段缺失或类型不匹配
# 3. 返回了额外的文本内容

print("\n" + "=" * 60)
print("3.2 不符合要求的情况 - 解析失败示例")
print("=" * 60)

# 定义一个Pydantic模型，用于提取电影信息
class MovieInfo(BaseModel):
    """电影信息模型"""
    title: str = Field(description="电影名称")
    director: str = Field(description="导演姓名")
    year: int = Field(description="上映年份")
    rating: float = Field(description="评分，0-10之间")
    genres: List[str] = Field(description="电影类型列表")

# 创建PydanticOutputParser
pydantic_parser = PydanticOutputParser(pydantic_object=MovieInfo)

# 获取格式指令，用于告诉模型如何输出
format_instructions = pydantic_parser.get_format_instructions()
print(f"\n格式指令：\n{format_instructions}")

# 模拟一个格式错误的输出
print("\n模拟格式错误的输出：")
print("-" * 40)

# 错误示例1：使用单引号而不是双引号（JSON标准要求双引号）
misformatted_output_1 = "{'title': '肖申克的救赎', 'director': '弗兰克·德拉邦特', 'year': 1994, 'rating': 9.7, 'genres': ['剧情', '犯罪']}"

print(f"错误输出1（单引号）: {misformatted_output_1}")
try:
    result = pydantic_parser.parse(misformatted_output_1)
    print(f"解析成功: {result}")
except Exception as e:
    print(f"❌ 解析失败: {type(e).__name__} - {str(e)[:100]}...")

# 错误示例2：字段类型错误（年份应该是数字，但返回了字符串）
misformatted_output_2 = '{"title": "阿甘正传", "director": "罗伯特·泽米吉斯", "year": "1994", "rating": 9.5, "genres": ["剧情", "爱情"]}'

print(f"\n错误输出2（类型错误）: {misformatted_output_2}")
try:
    result = pydantic_parser.parse(misformatted_output_2)
    print(f"解析成功: {result}")
except Exception as e:
    print(f"❌ 解析失败: {type(e).__name__} - {str(e)[:100]}...")

# 错误示例3：缺少必填字段
misformatted_output_3 = '{"title": "泰坦尼克号", "director": "詹姆斯·卡梅隆", "year": 1997, "rating": 9.4}'

print(f"\n错误输出3（缺少字段）: {misformatted_output_3}")
try:
    result = pydantic_parser.parse(misformatted_output_3)
    print(f"解析成功: {result}")
except Exception as e:
    print(f"❌ 解析失败: {type(e).__name__} - {str(e)[:100]}...")

# ==============================================
# 3.3 Auto-Fixing Parser自动修复
# ==============================================
# OutputFixingParser可以在解析失败时，自动调用大模型来修复格式错误。
# 工作原理：
# 1. 尝试使用原始解析器解析输出
# 2. 如果失败，将错误信息和原始输出一起发给大模型
# 3. 要求大模型修复格式问题
# 4. 使用修复后的输出再次尝试解析

print("\n" + "=" * 60)
print("3.3 Auto-Fixing Parser - 自动修复格式错误")
print("=" * 60)
print("""
OutputFixingParser是一个强大的容错工具：
- 它包装了一个基础解析器（如PydanticOutputParser）
- 当解析失败时，自动调用LLM来修复格式错误
- 支持配置最大重试次数
- 适用于生产环境中的鲁棒性需求

使用场景：
1. 大模型偶尔输出格式错误的JSON
2. 需要保证程序不会因为格式问题崩溃
3. 希望自动处理而非人工干预
""")

# 创建OutputFixingParser
fixing_parser = OutputFixingParser.from_llm(
    parser=pydantic_parser,    # 基础解析器
    llm=llm                    # 用于修复的大模型
)

# 测试自动修复功能
print("\n测试自动修复功能：")
print("-" * 40)

# 使用之前格式错误的输出进行测试
test_cases = [
    ("单引号JSON", misformatted_output_1),
    ("类型错误", misformatted_output_2),
    ("缺少字段", misformatted_output_3)
]

for name, test_output in test_cases:
    print(f"\n📝 测试用例: {name}")
    print(f"原始输出: {test_output}")
    try:
        # 使用OutputFixingParser解析
        result = fixing_parser.parse(test_output)
        print(f"✅ 修复成功!")
        print(f"  title: {result.title}")
        print(f"  director: {result.director}")
        print(f"  year: {result.year}")
        print(f"  rating: {result.rating}")
        print(f"  genres: {result.genres}")
    except Exception as e:
        print(f"❌ 修复失败: {str(e)[:100]}...")

# ==============================================
# 3.4 完整代码 - 实战示例
# ==============================================
# 综合示例：从用户输入中提取结构化的餐厅推荐信息

print("\n" + "=" * 60)
print("3.4 完整代码 - 餐厅推荐信息提取实战")
print("=" * 60)

# 定义餐厅信息模型
class RestaurantRecommendation(BaseModel):
    """餐厅推荐信息模型"""
    name: str = Field(description="餐厅名称")
    cuisine: str = Field(description="菜系，如中餐、西餐、日料等")
    price_range: str = Field(description="价格区间，如人均50元以下、50-100元、100-200元、200元以上")
    rating: float = Field(description="推荐指数，0-5星")
    reasons: List[str] = Field(description="推荐理由列表")

# 创建解析器
restaurant_parser = PydanticOutputParser(pydantic_object=RestaurantRecommendation)
restaurant_fixing_parser = OutputFixingParser.from_llm(
    parser=restaurant_parser,
    llm=llm
)

# 构建Prompt，包含格式指令
system_prompt = SystemMessage(content=f"""
你是一个美食推荐助手。请根据用户的需求，推荐一家合适的餐厅。

请严格按照以下JSON格式输出：
{restaurant_parser.get_format_instructions()}

注意：
1. 只输出JSON格式，不要包含其他文字
2. rating字段必须是数字（0-5）
3. reasons字段必须是数组
""")

# 用户输入
user_input = "我想找一家适合约会的餐厅，环境要好，价格适中，最好是西餐"

# 构建消息
messages = [
    system_prompt,
    HumanMessage(content=user_input)
]

# 调用LLM
print(f"\n用户需求: {user_input}")
print("\n正在调用大模型获取推荐...")

# 获取原始输出
raw_response = llm.invoke(messages)
raw_content = raw_response.content
print(f"\n大模型原始输出:\n{raw_content}")

# 使用自动修复解析器解析
print("\n正在解析输出...")
try:
    result = restaurant_fixing_parser.parse(raw_content)
    print("\n🎉 解析成功！餐厅推荐信息如下：")
    print("-" * 40)
    print(f"🍽️ 餐厅名称: {result.name}")
    print(f"🌍 菜系: {result.cuisine}")
    print(f"💰 价格区间: {result.price_range}")
    print(f"⭐ 推荐指数: {'⭐' * int(result.rating)}{'☆' * (5 - int(result.rating))} ({result.rating}/5)")
    print(f"\n📋 推荐理由:")
    for i, reason in enumerate(result.reasons, 1):
        print(f"  {i}. {reason}")
except Exception as e:
    print(f"\n❌ 解析失败: {str(e)}")

# ==============================================
# 4. 输出解析器总结
# ==============================================
print("\n" + "=" * 60)
print("4. 输出解析器总结")
print("=" * 60)
print("""
┌────────────────────────┬──────────────────────────────────────────────┐
│     解析器类型          │              用途说明                        │
├────────────────────────┼──────────────────────────────────────────────┤
│ StrOutputParser        │ 返回原始字符串输出（最简单）                  │
│ PydanticOutputParser   │ 解析为Pydantic模型，支持类型校验              │
│ OutputFixingParser     │ 自动修复格式错误，提升鲁棒性                  │
│ SimpleJsonOutputParser │ 解析简单JSON为字典                          │
│ CommaSeparatedList     │ 解析逗号分隔的列表                          │
│                        │                                            │
└────────────────────────┴──────────────────────────────────────────────┘

核心要点：
1. PydanticOutputParser是最常用的结构化解析器
2. OutputFixingParser是生产环境的必备容错工具
3. 使用get_format_instructions()生成格式指令，放入Prompt中
4. LCEL语法（|）可以方便地组装Chain

工作流程：
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│  Prompt     │───▶│    LLM      │───▶│ OutputParser    │
│ (含格式指令) │    │  生成输出   │    │  校验并结构化   │
└─────────────┘    └─────────────┘    └─────────────────┘
                                              │
                                      ┌───────┴───────┐
                                      │               │
                                   成功            失败
                                      │               │
                              ┌───────▼───────┐  ┌────▼────┐
                              │ 返回结构化数据 │  │抛出异常 │
                              └───────────────┘  └─────────┘
                                      │
                                      ▼
                              OutputFixingParser介入
                              自动修复并重试
""")