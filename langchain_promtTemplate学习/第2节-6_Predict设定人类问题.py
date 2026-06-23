# LangChain Predict部分：大模型接口封装
# 本文件演示如何使用LangChain调用大模型，以及不同Message类型的用途

# 1. 加载环境变量
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# 2. 导入必要的模块
# ChatOpenAI: LangChain封装的OpenAI聊天模型客户端
from langchain_openai import ChatOpenAI
# Message类型：LangChain定义的消息结构，用于构建多轮对话
# HumanMessage: 人类用户发送的消息，对应ChatGPT API中的user角色
# AIMessage: AI助手回复的消息，对应ChatGPT API中的assistant角色
# SystemMessage: 系统消息，用于设定AI的角色和行为准则，对应ChatGPT API中的system角色
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 3. 初始化大模型实例
# 从环境变量获取API密钥和Base URL
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

# 创建ChatOpenAI实例，配置超时时间和重试次数
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=30.0,    # 请求超时时间，单位秒
    max_retries=2    # 请求失败时的重试次数
)

# ==============================================
# 4. 什么是HumanMessage？
# ==============================================
# HumanMessage是LangChain中表示"人类用户输入"的消息类型
# 它对应ChatGPT API中的"user"角色
# 在多轮对话中，所有来自用户的提问都应该使用HumanMessage

print("=" * 60)
print("4. 什么是HumanMessage？")
print("=" * 60)
print("""
HumanMessage说明：
- 表示人类用户发送的消息
- 对应ChatGPT API中的"user"角色
- 在多轮对话中，所有用户输入都应该使用HumanMessage
- 结构：HumanMessage(content="用户输入内容")
""")

# 示例：使用HumanMessage发送单条消息
human_msg = HumanMessage(content="你好，我是狂龙哥")
response = llm.invoke([human_msg])
print("\n示例1 - 单条HumanMessage调用：")
print(f"用户输入: {human_msg.content}")
print(f"AI回复: {response.content}")

# ==============================================
# 5. 多轮对话：组合多种Message类型
# ==============================================
print("\n" + "=" * 60)
print("5. 多轮对话：组合多种Message类型")
print("=" * 60)
print("""
多轮对话的消息顺序：
1. SystemMessage - 设定AI角色（可选但推荐）
2. HumanMessage - 用户第一轮提问
3. AIMessage - AI第一轮回复
4. HumanMessage - 用户第二轮提问
5. ...以此类推

注意：消息数组的顺序决定了对话上下文的顺序
""")

# 构建多轮对话的消息数组
messages = [
    # SystemMessage：设定AI的角色和行为准则，最先发送
    SystemMessage(content="你是一个热情、幽默的个人助理，名字叫小明"),
    # HumanMessage：用户的第一轮提问
    HumanMessage(content="你好，我叫狂龙哥"),
    # AIMessage：模拟AI的第一轮回复（用于构建对话历史）
    AIMessage(content="狂龙哥你好！很高兴认识你，有什么我可以帮你的吗？"),
    # HumanMessage：用户的第二轮提问
    HumanMessage(content="我是谁？")
]

# 使用消息数组调用模型
response = llm.invoke(messages)
print("\n示例2 - 多轮对话调用：")
print("对话历史：")
for msg in messages:
    msg_type = msg.__class__.__name__
    print(f"  [{msg_type}] {msg.content}")
print(f"\nAI回复: {response.content}")

# ==============================================
# 6. 批量调用：一次处理多条消息
# ==============================================
print("\n" + "=" * 60)
print("6. 批量调用：一次处理多条消息")
print("=" * 60)
print("""
使用batch方法可以一次发送多个独立的请求
适用于需要并行处理多个不相关问题的场景
""")

# 准备多个独立的问题
batch_messages = [
    [HumanMessage(content="什么是人工智能？")],
    [HumanMessage(content="Python有哪些常用的数据结构？")],
    [HumanMessage(content="如何学习编程？")]
]

# 使用batch方法批量调用
responses = llm.batch(batch_messages)
print("\n示例3 - 批量调用结果：")
for i, (msgs, resp) in enumerate(zip(batch_messages, responses), 1):
    print(f"\n问题{i}: {msgs[0].content}")
    print(f"回答{i}: {resp.content}")

# ==============================================
# 7. 流式调用：实时获取回复
# ==============================================
print("\n" + "=" * 60)
print("7. 流式调用：实时获取回复")
print("=" * 60)
print("""
使用stream方法可以实时获取AI的回复内容
适用于需要显示打字效果或实时处理的场景
""")

# 使用stream方法流式调用
stream_messages = [
    SystemMessage(content="请用简洁的语言回答"),
    HumanMessage(content="什么是LangChain？")
]

print("\n示例4 - 流式调用结果：")
print("AI回复(流式输出): ", end="")
for chunk in llm.stream(stream_messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)
print()

# ==============================================
# 8. Message类型总结
# ==============================================
print("\n" + "=" * 60)
print("8. Message类型总结")
print("=" * 60)
print("""
┌─────────────────┬──────────────────────────────────────────────┐
│   Message类型    │              用途说明                        │
├─────────────────┼──────────────────────────────────────────────┤
│ SystemMessage   │ 设定AI的角色、行为准则、系统指令              │
│ HumanMessage    │ 人类用户的输入消息（最常用）                  │
│ AIMessage       │ AI助手的回复消息（用于构建对话历史）          │
└─────────────────┴──────────────────────────────────────────────┘

核心要点：
1. HumanMessage是最常用的消息类型，代表用户输入
2. SystemMessage用于设定上下文环境，影响AI的行为模式
3. AIMessage用于多轮对话中携带历史回复
4. 消息的顺序非常重要，决定了模型理解的对话上下文
""")
