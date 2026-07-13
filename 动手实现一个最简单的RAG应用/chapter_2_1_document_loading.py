#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2.1 文档加载与分块

本模块实现RAG流程的第一步：将原始文档加载并切分成小块。
文档分块是RAG的关键步骤，合适的分块大小直接影响检索效果和回答质量。

主要功能：
1. 加载本地文档文件（支持txt、md等文本格式）
2. 使用LangChain的文本分割器进行分块
3. 为每个文本块添加元数据
4. 输出分块结果供后续步骤使用
"""

import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document


def load_document(file_path: str) -> str:
    """
    加载文档文件内容
    
    Args:
        file_path: 文档文件的绝对路径
        
    Returns:
        文档的文本内容
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式不支持
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文档文件不存在: {file_path}")
    
    # 获取文件扩展名
    _, ext = os.path.splitext(file_path)
    
    # 支持的文件格式
    supported_extensions = ['.txt', '.md', '.mdx']
    
    if ext.lower() not in supported_extensions:
        raise ValueError(f"不支持的文件格式: {ext}，支持的格式: {supported_extensions}")
    
    # 读取文件内容
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except UnicodeDecodeError:
        # 尝试其他编码
        with open(file_path, 'r', encoding='gbk') as f:
            content = f.read()
        return content


def split_document(content: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    将文档内容切分成小块
    
    Args:
        content: 文档的文本内容
        chunk_size: 每个文本块的最大字符数
        chunk_overlap: 相邻文本块之间的重叠字符数
        
    Returns:
        Document对象列表，每个对象包含一个文本块及其元数据
        
    关键参数说明：
        - chunk_size: 控制每个块的大小，过大可能导致信息过载，过小可能丢失上下文
        - chunk_overlap: 确保相邻块之间有重叠，保持上下文连贯性
    """
    # 创建文本分割器
    # CharacterTextSplitter是最基础的分割器，按字符数分割
    # 其他可选分割器：
    # - RecursiveCharacterTextSplitter: 按多种分隔符递归分割（推荐）
    # - TokenTextSplitter: 按token数分割
    # - MarkdownTextSplitter: 专门处理Markdown格式
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,       # 每个块的最大字符数
        chunk_overlap=chunk_overlap, # 相邻块的重叠字符数
        separator="\n",              # 分割符，这里按换行符分割
        length_function=len,         # 长度计算函数
    )
    
    # 将文本分割成块
    # 注意：create_documents方法接收文本列表，返回Document对象列表
    chunks = text_splitter.create_documents([content])
    
    # 为每个块添加元数据
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["total_chunks"] = len(chunks)
    
    return chunks


def main():
    """
    主函数：演示文档加载与分块流程
    """
    print("=" * 60)
    print("2.1 文档加载与分块")
    print("=" * 60)
    
    # 示例文档路径
    doc_path = os.path.join(os.path.dirname(__file__), "docs", "example.txt")
    
    try:
        # 步骤1：加载文档
        print("\n[步骤1] 加载文档...")
        content = load_document(doc_path)
        print(f"文档总字符数: {len(content)}")
        print(f"文档前100字符: {content[:100]}...")
        
        # 步骤2：分块处理
        print("\n[步骤2] 文档分块...")
        chunks = split_document(content, chunk_size=300, chunk_overlap=30)
        print(f"分割后的块数: {len(chunks)}")
        
        # 步骤3：输出分块结果
        print("\n[步骤3] 分块结果展示:")
        for i, chunk in enumerate(chunks):
            print(f"\n--- 块 {i+1}/{len(chunks)} ---")
            print(f"字符数: {len(chunk.page_content)}")
            print(f"元数据: {chunk.metadata}")
            print(f"内容预览: {chunk.page_content[:100]}...")
            
    except Exception as e:
        print(f"\n❌ 出错了: {str(e)}")
        print("\n💡 常见问题排查:")
        print("  1. 检查文档路径是否正确")
        print("  2. 确保文档编码为UTF-8或GBK")
        print("  3. 确认文件格式是txt或md")


if __name__ == "__main__":
    main()
