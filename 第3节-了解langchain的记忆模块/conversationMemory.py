# ============================================================================
# LangChain 记忆模块（Memory）详解
# ============================================================================
# 为什么要用 Memory？
#   - 大模型本身是"无状态"的：每次请求都是独立的，不会记住之前的对话
#   - Memory 就是为了让 AI "记住"对话历史，实现多轮连贯对话
#
# Memory 工作流程：
#   用户输入 → 读取历史记忆 → 组装Prompt → 调用大模型 → 回复存入记忆 → 供下次使用
# ============================================================================

import sys
import os

# ============================================================================
# 第一节：前置准备 - 添加项目根目录到 Python 路径
# ============================================================================
# 因为这个脚本在子目录中，需要添加父目录才能导入 llm_manager
# __file__ 是当前脚本的路径，os.path.dirname 获取目录，os.path.pardir 获取父目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============================================================================
# 第二节：导入必要的模块
# ============================================================================

# 控制是否运行需要调用 LLM 的示例（如果 API 余额不足，设为 False）
RUN_LLM_EXAMPLES = True  # 设为 False 可跳过需要 LLM 的示例
from langchain_classic.memory import (
    ConversationBufferMemory,          # 完整缓冲记忆（全量保存对话）
    ConversationBufferWindowMemory,     # 滑动窗口记忆（只保留最近K轮）
    ConversationSummaryBufferMemory     # 摘要缓冲记忆（近期全量+远期摘要）
)
from langchain_community.chat_message_histories import ChatMessageHistory  # 底层聊天历史
from langchain_classic.chains.conversation.base import ConversationChain  # 对话链（整合LLM和Memory）
from llm_manager import LLMManager  # 项目自定义的LLM管理器


# ============================================================================
# 第二节：底层基础 - ChatMessageHistory
# ============================================================================
# ChatMessageHistory 是最底层的消息存储类，其他 Memory 都是基于它构建的
# 作用：直接管理消息列表，可以手动添加用户消息和AI回复
#
# 适用场景：
#   - 需要完全控制消息历史的场景
#   - 作为其他高级 Memory 的底层存储
# ============================================================================
print("=" * 60)
print("【示例1】ChatMessageHistory - 最底层的消息存储")
print("=" * 60)

# 创建一个空的聊天历史对象
history = ChatMessageHistory()

# 添加用户消息
history.add_user_message("你好，我叫张三")
print(f"添加用户消息后: {history.messages}")

# 添加AI回复
history.add_ai_message("你好张三！很高兴认识你。")
print(f"添加AI回复后: {history.messages}")

# 打印消息列表（可以看到是 HumanMessage 和 AIMessage 对象）
print(f"\n消息类型分析:")
for msg in history.messages:
    print(f"  - {type(msg).__name__}: {msg.content}")


# ============================================================================
# 第三节：ConversationBufferMemory - 完整缓冲记忆
# ============================================================================
# 原理：全量保存所有对话历史，无压缩、无截断
#
# 优点：
#   ✅ 上下文100%完整，不会丢失任何信息
#   ✅ 逻辑最连贯，模型能访问所有历史上下文
#
# 缺点：
#   ❌ Token数量随对话轮次线性增长
#   ❌ 长对话容易超出模型的上下文限制
#   ❌ 无持久化，服务重启后记忆丢失
#
# 适用场景：
#   - 短轮次对话（建议不超过10轮）
#   - 对上下文完整性要求极高的应用
#   - 开发调试阶段
# ============================================================================
print("\n" + "=" * 60)
print("【示例2】ConversationBufferMemory - 完整缓冲记忆")
print("=" * 60)

# 创建记忆对象
memory = ConversationBufferMemory()

# 保存上下文：第一个参数是用户输入，第二个参数是AI回复
memory.save_context({"input": "你好啊"}, {"output": "你好我是你的狂龙哥！"})
print(f"第1轮对话后记忆:\n  {memory.load_memory_variables({})}")

# 再来一轮对话
memory.save_context({"input": "嘿，我又来了，你再好啊"}, {"output": "知道了，我是狂龙哥，别再BB了"})
print(f"\n第2轮对话后记忆:\n  {memory.load_memory_variables({})}")

# 使用 save_context 后，history 中会自动添加消息
print(f"\nChatMessageHistory 中的消息列表:")
for msg in memory.chat_memory.messages:
    print(f"  - {msg.type}: {msg.content}")


# ============================================================================
# 第四节：ConversationBufferWindowMemory - 滑动窗口记忆
# ============================================================================
# 原理：只保留最近 N 轮对话，超出窗口的历史自动丢弃
#
# 核心参数：
#   k: 窗口大小，即保留最近几轮对话（默认5）
#
# 优点：
#   ✅ Token消耗稳定可控，不会无限增长
#   ✅ 实现简单，性能好
#
# 缺点：
#   ❌ 会丢失早期历史信息
#
# 适用场景：
#   - 日常聊天、通用智能体
#   - 长轮次对话（生产环境首选）
#   - 客服系统等轮次相对固定的场景
# ============================================================================
print("\n" + "=" * 60)
print("【示例3】ConversationBufferWindowMemory - 滑动窗口记忆（k=2）")
print("=" * 60)

# 创建滑动窗口记忆，只保留最近2轮对话
memory_window = ConversationBufferWindowMemory(k=2)

# 模拟一个较长的对话
memory_window.save_context({"input": "第1轮：我是张三"}, {"output": "第1轮：你好张三"})
memory_window.save_context({"input": "第2轮：我25岁"}, {"output": "第2轮：知道了，25岁"})
memory_window.save_context({"input": "第3轮：我在深圳"}, {"output": "第3轮：深圳是个好地方"})
memory_window.save_context({"input": "第4轮：我是程序员"}, {"output": "第4轮：程序员辛苦了"})

print("进行4轮对话后，只保留最近2轮:")
print(f"  {memory_window.load_memory_variables({})}")

# 可以查看被丢弃的历史
print(f"\n被丢弃的轮次:")
try:
    print(f"  {memory_window.chat_memory.messages}")
except Exception as e:
    print(f"  已被丢弃，无法查看（这是正常的）")


# ============================================================================
# 第五节：ConversationSummaryBufferMemory - 摘要缓冲记忆（推荐）
# ============================================================================
# 原理：结合"近期全量"和"远期摘要"的分层存储机制
#
# 核心参数：
#   max_token_limit: 总Token容量上限（达到后触发摘要压缩）
#
# 工作机制：
#   - 短期记忆区：存储最近几轮的完整原文，保留细节
#   - 长期记忆区：将早期对话压缩为结构化摘要，存储关键信息
#   - 动态压缩：当总Token接近上限时，自动将最早轮次压缩为摘要
#
# 优点：
#   ✅ 近详远略，保留近期细节的同时不丢失早期主旨
#   ✅ 自动压缩，节省约70%空间
#   ✅ 适合大多数生产场景（官方推荐）
#
# 缺点：
#   ❌ 摘要生成会消耗额外Token
#   ❌ 可能丢失一些细节信息
#
# 适用场景：
#   - 技术问答助手（需要保留精确技术细节）
#   - 长对话应用
#   - 大多数生产环境首选方案
# ============================================================================
print("\n" + "=" * 60)
print("【示例4】ConversationSummaryBufferMemory - 摘要缓冲记忆")
print("=" * 60)

if RUN_LLM_EXAMPLES:
    try:
        # 创建摘要缓冲记忆，设置最大Token限制
        llm = LLMManager().llm
        memory_summary = ConversationSummaryBufferMemory(
            llm=llm,  # 必须传入LLM，用于生成摘要
            max_token_limit=100  # 达到100 Token后触发摘要
        )

        # 添加一些对话
        memory_summary.save_context(
            {"input": "我叫张三，是一名Python后端开发工程师"},
            {"output": "你好张三！Python后端开发很有前景，你在哪个城市工作？"}
        )
        memory_summary.save_context(
            {"input": "我在深圳，在一家互联网公司做后端开发"},
            {"output": "深圳的互联网行业很发达，你们公司主要做什么产品？"}
        )
        memory_summary.save_context(
            {"input": "我们公司做社交产品的，用户量还挺大的"},
            {"output": "社交产品很有意思！用户量大概多少？"}
        )

        # 查看记忆内容
        memory_vars = memory_summary.load_memory_variables({})
        print("记忆内容（包含摘要和缓冲）：")
        print(f"  {memory_vars}")

        # 打印缓冲区内容（ConversationSummaryBufferMemory 的 buffer 是原始消息列表）
        print(f"\n缓冲区内容（原始消息列表）：")
        if hasattr(memory_summary, 'chat_memory') and memory_summary.chat_memory:
            # 使用 chat_memory 获取消息对象
            for msg in memory_summary.chat_memory.messages:
                role = "用户" if msg.type == "human" else "AI"
                content = msg.content if hasattr(msg, 'content') else str(msg)
                print(f"  [{role}] {content[:50]}...")
        else:
            print(f"  {memory_summary.buffer}")
    except Exception as e:
        print(f"⚠️  示例4运行失败（可能API余额不足）: {str(e)[:100]}")
        print("   将 RUN_LLM_EXAMPLES 设为 False 可跳过需要 LLM 的示例")
else:
    print("⏭️  已跳过（RUN_LLM_EXAMPLES = False）")
    print("   ConversationSummaryBufferMemory 需要 LLM 生成摘要")


# ============================================================================
# 第六节：完整示例 - 使用 ConversationChain 整合 LLM 和 Memory
# ============================================================================
# ConversationChain 是 LangChain 提供的对话链封装
# 作用：自动将 Memory 中的历史与新的用户输入整合，发送给 LLM
#
# 工作流程：
#   用户输入 → Memory加载历史 → 整合成完整Prompt → LLM推理 → 回复存入Memory
# ============================================================================
print("\n" + "=" * 60)
print("【示例5】ConversationChain - 完整的带记忆对话")
print("=" * 60)

if RUN_LLM_EXAMPLES:
    try:
        # 创建带完整缓冲记忆的对话链
        conversation_memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=llm,
            memory=conversation_memory,
            verbose=False  # 设为True可以看到完整的Prompt构建过程
        )

        # 第一轮对话
        print("\n--- 第1轮对话 ---")
        response1 = conversation.invoke({"input": "你好，我叫张三"})
        print(f"AI回复: {response1['response']}")

        # 第二轮对话（AI会自动记住"张三"这个名字）
        print("\n--- 第2轮对话 ---")
        response2 = conversation.invoke({"input": "你还记得我叫什么吗？"})
        print(f"AI回复: {response2['response']}")

        # 第三轮对话
        print("\n--- 第3轮对话 ---")
        response3 = conversation.invoke({"input": "我是做什么工作的？"})
        print(f"AI回复: {response3['response']}")

        # 查看完整的对话历史
        print("\n--- 查看记忆内容 ---")
        print(f"记忆: {conversation_memory.load_memory_variables({})}")
    except Exception as e:
        print(f"⚠️  示例5运行失败（可能API余额不足）: {str(e)[:100]}")
        print("   将 RUN_LLM_EXAMPLES 设为 False 可跳过需要 LLM 的示例")
else:
    print("⏭️  已跳过（RUN_LLM_EXAMPLES = False）")
    print("   ConversationChain 需要 LLM 进行对话")


# ============================================================================
# 第七节：滑动窗口 + ConversationChain 实战
# ============================================================================
print("\n" + "=" * 60)
print("【示例6】滑动窗口记忆实战（客服场景）")
print("=" * 60)

if RUN_LLM_EXAMPLES:
    try:
        # 模拟客服场景，只保留最近3轮对话
        customer_memory = ConversationBufferWindowMemory(k=3)
        customer_conversation = ConversationChain(
            llm=llm,
            memory=customer_memory
        )

        # 模拟一段较长的客服对话
        dialogue_history = [
            "我想咨询一下你们的会员服务",
            "我们有月度会员和年度会员，月度99元，年度799元",
            "年度会员有什么优惠吗？",
            "年度会员可以省约200元，相当于8个月的价格，而且有专属客服",
            "听起来不错，我考虑一下",
            "好的，有什么问题随时联系我",
            "那我先了解一下退款政策",
            "我们的退款政策是7天内无理由退款，超过7天按剩余时长折算"
        ]

        print("开始一段8轮的客服对话（窗口大小k=3，只会保留最近3轮）:\n")

        for i, user_input in enumerate(dialogue_history, 1):
            response = customer_conversation.invoke({"input": user_input})
            print(f"轮次{i} | 用户: {user_input[:20]}...")
            print(f"       | AI: {response['response'][:40]}...")
            print()

        # 对话结束后，看看记忆中还剩什么
        final_memory = customer_memory.load_memory_variables({})
        print("对话结束后，只保留最近3轮（k=3）:")
        print(f"  {final_memory}")
    except Exception as e:
        print(f"⚠️  示例6运行失败（可能API余额不足）: {str(e)[:100]}")
        print("   将 RUN_LLM_EXAMPLES 设为 False 可跳过需要 LLM 的示例")
else:
    print("⏭️  已跳过（RUN_LLM_EXAMPLES = False）")
    print("   ConversationChain 需要 LLM 进行对话")


# ============================================================================
# 总结：Memory 类型对比
# ============================================================================
print("\n" + "=" * 60)
print("【总结】LangChain Memory 类型对比")
print("=" * 60)
print("""
| 类型                    | 存储内容        | 控制方式    | 优点                  | 缺点                    | 适用场景              |
|------------------------|---------------|-----------|---------------------|-----------------------|---------------------|
| ChatMessageHistory     | 原始消息列表     | 手动管理    | 完全可控              | 需要自己管理逻辑           | 底层存储/自定义场景     |
| ConversationBufferMemory | 全部原始消息    | 无（手动截断）| 信息完整              | Token线性增长           | 短对话/调试           |
| ConversationBufferWindowMemory | 最近K轮对话   | 固定轮数K   | 长度可控              | 早期信息被丢弃           | 聊天室/客服           |
| ConversationSummaryBufferMemory | 近期全量+远期摘要 | Token上限  | 近详远略，自动压缩      | 摘要消耗Token           | 大多数生产场景（推荐）   |
""")

print("""
【选型建议】
1. 短对话（<10轮）→ ConversationBufferMemory，简单直接
2. 日常聊天/客服 → ConversationBufferWindowMemory，控制成本
3. 长对话/生产环境 → ConversationSummaryBufferMemory，平衡效果和成本
4. 需要完全自定义 → 直接使用 ChatMessageHistory

【注意事项】
⚠️ 传统 Memory 类在 LangChain 0.3+ 已弃用，将在 2.0 中移除
⚠️ 官方推荐使用 LangGraph + Checkpoint 作为新的状态管理方案
⚠️ 所有 Memory 都是内存存储，服务重启会丢失，需要持久化请配合数据库使用
""")
