#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4. 使用大模型得到答案

本模块实现RAG流程的第四步：使用大语言模型基于构建好的Prompt生成回答。

教程目录对应：
- 4.1 封装海外领先AI厂商接口
- 4.2 组装Prompt
- 4.3 使用大模型得到答案

主要功能：
1. 封装大模型API调用（支持OpenAI兼容接口）
2. 将检索到的文档与用户问题组装成完整Prompt
3. 调用大模型生成回答
4. 处理API调用中的常见错误和异常
"""

import os
import sys
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
_ = load_dotenv(find_dotenv())

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.documents import Document
except ImportError as e:
    print(f"❌ 导入依赖失败: {e}")
    print("💡 请安装所需依赖: pip install langchain-openai langchain-core")
    sys.exit(1)


def init_llm(model_name: str = "deepseek-chat", temperature: float = 0.7) -> ChatOpenAI:
    """
    4.1 封装海外领先AI厂商接口
    
    初始化大语言模型实例，支持OpenAI兼容的API接口
    
    Args:
        model_name: 模型名称，默认为deepseek-chat
        temperature: 温度参数，控制回答的随机性（0-1）
        
    Returns:
        ChatOpenAI实例
        
    支持的模型：
        - deepseek-chat: DeepSeek对话模型
        - gpt-3.5-turbo: OpenAI GPT-3.5
        - gpt-4: OpenAI GPT-4
        - 其他OpenAI兼容的模型
        
    配置说明：
        需要在.env文件中设置：
        OPENAI_API_KEY=你的API密钥
        OPENAI_BASE_URL=API基础地址（如https://api.deepseek.com/v1）
        
    踩坑提示：
        错误：ModuleNotFoundError: No module named 'langchain_openai'
        原因：LangChain版本更新后，OpenAI集成从langchain移到了langchain_openai包
        解决：pip install langchain-openai
    """
    try:
        llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=temperature,
            timeout=30,
            max_retries=2,
        )
        print(f"✅ 大模型初始化成功: {model_name}")
        return llm
    except Exception as e:
        print(f"❌ 大模型初始化失败: {e}")
        print("\n💡 常见问题排查:")
        print("   1. 检查.env文件中的OPENAI_API_KEY是否设置正确")
        print("   2. 检查OPENAI_BASE_URL是否指向正确的API地址")
        print("   3. 确保API Key有足够的余额和权限")
        raise


def assemble_prompt(question: str, documents: list, prompt_template) -> str:
    """
    4.2 组装Prompt
    
    将用户问题和检索到的文档组装成完整的Prompt
    
    Args:
        question: 用户问题
        documents: 检索到的文档列表
        prompt_template: PromptTemplate实例
        
    Returns:
        完整的Prompt字符串
        
    流程：
        1. 将文档列表格式化为上下文字符串
        2. 将上下文和问题注入Prompt模板
        3. 返回组装好的Prompt
    """
    # 格式化文档内容
    context_parts = []
    for i, doc in enumerate(documents):
        content = doc.page_content.strip()
        if content:
            context_parts.append(f"【文档{i+1}】\n{content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    if not context:
        context = "（未检索到相关文档）"
    
    # 组装Prompt
    prompt = prompt_template.format(
        context=context,
        question=question,
    )
    
    return prompt


def generate_answer(llm, prompt: str) -> str:
    """
    4.3 使用大模型得到答案
    
    调用大模型API生成回答
    
    Args:
        llm: ChatOpenAI实例
        prompt: 完整的Prompt字符串
        
    Returns:
        模型生成的回答文本
        
    说明：
        使用StrOutputParser解析模型输出，将ChatMessage转换为字符串
    """
    # 创建输出解析器
    output_parser = StrOutputParser()
    
    # 构建链式调用：prompt -> llm -> output_parser
    # 注意：这里简化了流程，实际生产中可以使用LangChain的Chain API
    response = llm.invoke(prompt)
    
    # 解析输出
    answer = output_parser.invoke(response)
    
    return answer.strip()


def rag_pipeline(llm, prompt_template, question: str, retriever) -> dict:
    """
    RAG完整流程：检索 + 生成
    
    Args:
        llm: ChatOpenAI实例
        prompt_template: PromptTemplate实例
        question: 用户问题
        retriever: 检索器对象
        
    Returns:
        包含回答和相关文档的字典
        
    流程：
        1. 使用检索器获取相关文档
        2. 组装Prompt
        3. 调用大模型生成回答
        4. 返回结果
    """
    # 步骤1：检索相关文档
    print("\n[1/3] 检索相关文档...")
    retrieved_docs = retriever.invoke(question)
    print(f"检索到 {len(retrieved_docs)} 个相关文档")
    
    # 步骤2：组装Prompt
    print("\n[2/3] 组装Prompt...")
    prompt = assemble_prompt(question, retrieved_docs, prompt_template)
    
    # 步骤3：生成回答
    print("\n[3/3] 调用大模型生成回答...")
    answer = generate_answer(llm, prompt)
    
    return {
        "question": question,
        "answer": answer,
        "retrieved_documents": retrieved_docs,
        "prompt": prompt,
    }


def main():
    """
    主函数：演示使用大模型生成答案的完整流程
    """
    print("=" * 60)
    print("4. 使用大模型得到答案")
    print("=" * 60)
    
    # 添加上级目录到Python路径
    sys.path.insert(0, os.path.dirname(__file__))
    
    try:
        from chapter_3_prompt_template import create_rag_prompt
        from chapter_2_2_vector_store import load_vector_store, create_retriever, create_embeddings
        
        # 步骤1：初始化大模型
        print("\n[步骤1] 初始化大模型...")
        llm = init_llm(model_name="deepseek-chat", temperature=0.3)
        
        # 步骤2：加载向量数据库和检索器
        print("\n[步骤2] 加载向量数据库和检索器...")
        # 重要：FAISS的C++底层无法处理中文路径，必须使用纯ASCII路径
        persist_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")
        print(f"Vector store path: {persist_path}")
        embeddings = create_embeddings()
        
        try:
            vector_store = load_vector_store(persist_path, embeddings)
        except FileNotFoundError:
            print("⚠️ 向量数据库不存在，先创建...")
            from chapter_2_1_document_loading import load_document, split_document
            doc_path = os.path.join(os.path.dirname(__file__), "docs", "example.txt")
            content = load_document(doc_path)
            chunks = split_document(content, chunk_size=300, chunk_overlap=30)
            from chapter_2_2_vector_store import create_vector_store
            vector_store = create_vector_store(chunks, embeddings, persist_path)
        
        retriever = create_retriever(vector_store, k=3)
        
        # 步骤3：创建Prompt模板
        print("\n[步骤3] 创建Prompt模板...")
        prompt_template = create_rag_prompt("detailed")
        
        # 步骤4：运行RAG流程
        print("\n[步骤4] 运行RAG流程...")
        
        # 测试问题1
        print("\n" + "-" * 50)
        print("测试问题1: 什么是RAG？")
        print("-" * 50)
        result1 = rag_pipeline(llm, prompt_template, "什么是RAG？", retriever)
        print(f"\n回答:\n{result1['answer']}")
        
        # 测试问题2
        print("\n" + "-" * 50)
        print("测试问题2: RAG的核心流程包括哪些步骤？")
        print("-" * 50)
        result2 = rag_pipeline(llm, prompt_template, "RAG的核心流程包括哪些步骤？", retriever)
        print(f"\n回答:\n{result2['answer']}")
        
        # 测试问题3（文档中可能没有相关信息）
        print("\n" + "-" * 50)
        print("测试问题3: 什么是深度学习？")
        print("-" * 50)
        result3 = rag_pipeline(llm, prompt_template, "什么是深度学习？", retriever)
        print(f"\n回答:\n{result3['answer']}")
        
    except Exception as e:
        print(f"\n❌ 出错了: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
