#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3. Prompt模板

本模块实现RAG流程的第三步：构建用于生成回答的Prompt模板。

Prompt模板是RAG的核心，它决定了如何将检索到的文档和用户问题
组合成一个有效的提示词，引导大模型生成准确的回答。

主要功能：
1. 设计RAG专用的Prompt模板
2. 实现文档内容的格式化
3. 将用户问题与检索到的文档结合
4. 提供多种模板样式供选择
"""

from langchain_core.prompts import PromptTemplate


def create_rag_prompt(template_type: str = "basic") -> PromptTemplate:
    """
    创建RAG专用的Prompt模板
    
    Args:
        template_type: 模板类型，可选值: basic, detailed, strict
        
    Returns:
        PromptTemplate实例
        
    模板说明：
        - basic: 基础模板，简洁明了
        - detailed: 详细模板，包含更多指令
        - strict: 严格模板，强调必须基于提供的文档回答
    """
    templates = {
        "basic": """
基于以下文档内容回答用户问题：

{context}

用户问题：{question}

请给出简洁准确的回答。
""",
        
        "detailed": """
你是一个专业的AI助手，擅长根据提供的文档内容回答问题。

请严格按照以下规则回答：
1. 优先使用提供的文档内容进行回答
2. 如果文档中没有相关信息，请明确说明"文档中未找到相关信息"
3. 回答要简洁明了，避免冗长
4. 如果需要举例，请使用文档中的内容

文档内容：
{context}

用户问题：{question}

回答：
""",
        
        "strict": """
任务：根据提供的参考文档回答用户问题。

规则：
- 你的回答必须完全基于以下参考文档
- 禁止使用任何外部知识
- 如果文档中没有相关信息，直接回答"无法回答"
- 回答要准确、简洁，不要添加无关内容

参考文档：
{context}

用户问题：{question}

答案：
"""
    }
    
    if template_type not in templates:
        template_type = "basic"
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=templates[template_type],
    )
    
    return prompt


def format_documents(documents: list) -> str:
    """
    将检索到的文档列表格式化为字符串
    
    Args:
        documents: Document对象列表
        
    Returns:
        格式化后的文档内容字符串
        
    说明：
        将多个文档合并成一个字符串，每个文档之间用分隔符隔开
        便于作为context注入到Prompt中
    """
    if not documents or len(documents) == 0:
        return ""
    
    formatted_docs = []
    for i, doc in enumerate(documents):
        content = doc.page_content.strip()
        if content:
            # 添加文档编号和来源信息
            source = doc.metadata.get("source", f"文档{i+1}")
            chunk_id = doc.metadata.get("chunk_id", i)
            formatted_doc = f"【文档{chunk_id+1}】\n{content}"
            formatted_docs.append(formatted_doc)
    
    # 使用分隔线连接多个文档
    return "\n\n---\n\n".join(formatted_docs)


def build_prompt(prompt_template: PromptTemplate, question: str, documents: list) -> str:
    """
    构建完整的Prompt
    
    Args:
        prompt_template: PromptTemplate实例
        question: 用户问题
        documents: 检索到的文档列表
        
    Returns:
        完整的Prompt字符串
        
    流程：
        1. 将文档列表格式化为字符串
        2. 将格式化后的文档和问题注入模板
        3. 返回完整的Prompt
    """
    # 格式化文档内容
    context = format_documents(documents)
    
    # 如果没有检索到文档，添加提示
    if not context:
        context = "（未检索到相关文档）"
    
    # 构建Prompt
    prompt = prompt_template.format(
        context=context,
        question=question,
    )
    
    return prompt


def main():
    """
    主函数：演示Prompt模板的使用
    """
    print("=" * 60)
    print("3. Prompt模板")
    print("=" * 60)
    
    # 创建示例文档（模拟检索到的结果）
    from langchain_core.documents import Document
    
    sample_docs = [
        Document(
            page_content="RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索和文本生成的AI技术。它通过从外部知识库中检索相关信息，增强大语言模型的回答能力。",
            metadata={"source": "rag_intro.txt", "chunk_id": 0}
        ),
        Document(
            page_content="RAG的核心流程包括：文档加载与分块、向量化与存储、检索相关文档、构建Prompt、生成回答。每个步骤都对最终的回答质量有重要影响。",
            metadata={"source": "rag_process.txt", "chunk_id": 1}
        ),
        Document(
            page_content="文档分块是RAG的关键步骤，常见的分块策略包括按字符数分割、按段落分割、按语义分割等。合适的分块大小通常在200-500字符之间。",
            metadata={"source": "chunking.txt", "chunk_id": 2}
        )
    ]
    
    # 用户问题
    user_question = "RAG是什么？它的核心流程包括哪些步骤？"
    
    # 步骤1：创建不同类型的模板
    print("\n[步骤1] 创建不同类型的Prompt模板")
    print("\n--- 基础模板 ---")
    basic_prompt = create_rag_prompt("basic")
    print(basic_prompt.template)
    
    print("\n--- 详细模板 ---")
    detailed_prompt = create_rag_prompt("detailed")
    print(detailed_prompt.template)
    
    print("\n--- 严格模板 ---")
    strict_prompt = create_rag_prompt("strict")
    print(strict_prompt.template)
    
    # 步骤2：构建完整Prompt
    print("\n" + "=" * 60)
    print("[步骤2] 构建完整Prompt")
    print("=" * 60)
    
    full_prompt = build_prompt(detailed_prompt, user_question, sample_docs)
    print(full_prompt)
    
    # 步骤3：展示格式化后的文档内容
    print("\n" + "=" * 60)
    print("[步骤3] 文档格式化示例")
    print("=" * 60)
    formatted_context = format_documents(sample_docs)
    print(formatted_context)


if __name__ == "__main__":
    main()
