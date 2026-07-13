#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2.2 创建向量数据库

本模块实现RAG流程的第二步：将分块后的文本向量化并存储到向量数据库中。

主要功能：
1. 使用嵌入模型将文本块转换为向量
2. 将向量存储到本地向量数据库（FAISS）
3. 创建检索器用于后续查询
4. 包含常见踩坑点的处理和说明

教程目录对应：
- 2.2.1 创建过程
- 2.2.2 运行结果
- 2.2.3 踩坑
  - 2.2.3.1 坑一：NoneType object is not iterable
  - 2.2.3.2 坑二：Number of embeddings 9 must match number of ids 10

使用方法：
    # 正常运行
    python chapter_2_2_vector_store.py
    
    # 演示坑一
    python chapter_2_2_vector_store.py --demo pit1
    
    # 演示坑二
    python chapter_2_2_vector_store.py --demo pit2
"""

import os
import sys
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("[TIP] Install dependencies: pip install langchain-openai langchain-community faiss-cpu")
    sys.exit(1)


def demo_pit1():
    """
    演示坑一：NoneType object is not iterable
    
    原因：API Key或Base URL配置错误导致模型初始化失败
    表现：调用embeddings.embed_documents()时返回None或连接失败
    解决：检查.env文件中的OPENAI_API_KEY和OPENAI_BASE_URL是否正确设置
    """
    print("\n" + "=" * 60)
    print("⚠️ 演示模式：坑一 - NoneType object is not iterable")
    print("=" * 60)
    print("\n场景：故意传入错误的API Key和Base URL，模拟配置错误")
    print("预期结果：连接失败，报错提示")
    
    try:
        # 故意使用错误的API Key触发坑一
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key="sk-WRONG_API_KEY_THIS_IS_A_DEMO",
            base_url="https://wrong-api-url.example.com/v1",
        )
        
        # 尝试调用嵌入模型，这会失败
        test_texts = ["测试文本"]
        result = embeddings.embed_documents(test_texts)
        
    except Exception as e:
        print(f"\n❌ 触发坑一成功！")
        print(f"错误信息: {type(e).__name__}: {e}")
        print("\n💡 踩坑提示 - 坑一：NoneType object is not iterable")
        print("   常见原因：")
        print("   1. .env文件中OPENAI_API_KEY未设置或设置错误")
        print("   2. .env文件中OPENAI_BASE_URL未设置或URL不正确")
        print("   3. API Key格式错误（如把key值当作环境变量名）")
        print("\n   正确示例:")
        print("   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx")
        print("   OPENAI_BASE_URL=https://api.deepseek.com/v1")
        print("\n   解决方法：")
        print("   1. 检查.env文件中的配置")
        print("   2. 使用os.getenv('OPENAI_API_KEY')读取密钥")
        print("   3. 确保网络可以访问API地址")
        return
    
    print("\n⚠️ 意外：坑一没有被触发！")
    return


def demo_pit2():
    """
    演示坑二：Number of embeddings 9 must match number of ids 10
    
    原因：文档列表中包含空文档或None
    表现：创建向量库时嵌入数量与文档ID数量不匹配
    解决：在创建向量库前过滤掉空文档
    """
    print("\n" + "=" * 60)
    print("⚠️ 演示模式：坑二 - Number of embeddings mismatch")
    print("=" * 60)
    print("\n场景：故意在文档列表中添加空文档")
    print("预期结果：FAISS创建失败，报错提示")
    
    from langchain_core.embeddings import Embeddings
    import numpy as np
    
    class MockEmbeddings(Embeddings):
        """模拟嵌入模型，用于演示坑二"""
        def embed_documents(self, texts):
            """故意对空文本返回空列表，模拟嵌入失败"""
            results = []
            for text in texts:
                if text.strip():
                    results.append(np.random.rand(1536).tolist())
                else:
                    # 关键：对空文本返回空列表，模拟真实嵌入模型的行为
                    pass
            return results
        
        def embed_query(self, text):
            return np.random.rand(1536).tolist()
    
    # 创建正常文档
    normal_docs = [
        Document(page_content="RAG是检索增强生成技术", metadata={"chunk_id": 0}),
        Document(page_content="文档分块是RAG的关键步骤", metadata={"chunk_id": 1}),
        Document(page_content="向量数据库用于存储和检索向量", metadata={"chunk_id": 2}),
    ]
    
    # 故意添加空文档触发坑二
    empty_doc = Document(page_content="", metadata={"chunk_id": 999})
    docs_with_empty = normal_docs + [empty_doc]
    
    print(f"\n正常文档数: {len(normal_docs)}")
    print(f"添加空文档后: {len(docs_with_empty)}")
    print(f"注意：最后一个文档内容为空字符串")
    print(f"\n模拟嵌入模型行为：对空文本返回空结果")
    
    try:
        # 使用模拟的嵌入模型
        embeddings = MockEmbeddings()
        
        # 故意不做过滤，直接创建向量库（触发坑二）
        vector_store = FAISS.from_documents(docs_with_empty, embeddings)
        
    except ValueError as e:
        if "Number of embeddings" in str(e) or "documents and embeddings expected to be equal length" in str(e):
            print(f"\n❌ 触发坑二成功！")
            print(f"错误信息: {e}")
            print("\n💡 踩坑提示 - 坑二：Number of embeddings 9 must match number of ids 10")
            print("   常见原因：")
            print("   1. 文档列表中包含空文档（page_content为空字符串）")
            print("   2. 文档列表中包含None元素")
            print("   3. 某些文档内容过短，嵌入模型返回异常")
            print("\n   解决方法：")
            print("   1. 在创建向量库前过滤掉空文档")
            print("   2. 检查文档分块逻辑，确保每个块都有内容")
            print("   3. 增加chunk_size最小值限制")
            print("\n   正确代码示例:")
            print("   valid_docs = [doc for doc in docs if doc.page_content.strip()]")
            print("   vector_store = FAISS.from_documents(valid_docs, embeddings)")
            return
        else:
            raise
    except Exception as e:
        print(f"\n❌ 发生其他错误: {type(e).__name__}: {e}")
        print("\n⚠️ 坑二演示可能因为FAISS版本差异未能触发")
        return
    
    print("\n⚠️ 意外：坑二没有被触发！")
    return


def create_embeddings(model_name: str = "text-embedding-3-small"):
    """创建嵌入模型实例"""
    try:
        embeddings = OpenAIEmbeddings(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        return embeddings
    except Exception as e:
        print(f"❌ 创建嵌入模型失败: {e}")
        print("\n💡 踩坑提示 - 坑一：NoneType object is not iterable")
        print("   常见原因：")
        print("   1. .env文件中OPENAI_API_KEY未设置或设置错误")
        print("   2. .env文件中OPENAI_BASE_URL未设置或URL不正确")
        print("   3. API Key格式错误（如把key值当作环境变量名）")
        print("\n   正确示例:")
        print("   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx")
        print("   OPENAI_BASE_URL=https://api.deepseek.com/v1")
        raise


def create_vector_store(documents: list, embeddings, persist_path: str = "./vector_store"):
    """创建向量数据库并持久化存储"""
    if not documents or len(documents) == 0:
        raise ValueError("文档列表为空，无法创建向量数据库")
    
    # 过滤空文档
    valid_docs = []
    for i, doc in enumerate(documents):
        if doc and doc.page_content and len(doc.page_content.strip()) > 0:
            valid_docs.append(doc)
        else:
            print(f"[WARNING] Skip empty or invalid document, index: {i}")
    
    print(f"\nOriginal docs: {len(documents)}, Valid docs: {len(valid_docs)}")
    
    if len(valid_docs) == 0:
        raise ValueError("所有文档都为空，无法创建向量数据库")
    
    try:
        vector_store = FAISS.from_documents(valid_docs, embeddings)
        
        os.makedirs(persist_path, exist_ok=True)
        vector_store.save_local(persist_path)
        print(f"[OK] Vector store saved to: {persist_path}")
        
        return vector_store
    except ValueError as e:
        if "Number of embeddings" in str(e):
            print(f"\n❌ {e}")
            print("\n💡 踩坑提示 - 坑二：Number of embeddings 9 must match number of ids 10")
            print("   常见原因：")
            print("   1. 文档列表中包含空文档（page_content为空字符串）")
            print("   2. 文档列表中包含None元素")
            print("   3. 某些文档内容过短，嵌入模型返回异常")
            print("\n   解决方法：")
            print("   1. 在创建向量库前过滤掉空文档")
            print("   2. 检查文档分块逻辑，确保每个块都有内容")
            print("   3. 增加chunk_size最小值限制")
        raise


def load_vector_store(persist_path: str = "./vector_store", embeddings=None):
    """加载已持久化的向量数据库"""
    if not os.path.exists(persist_path):
        raise FileNotFoundError(f"Vector store not found: {persist_path}")
    
    if embeddings is None:
        embeddings = create_embeddings()
    
    vector_store = FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
    print(f"[OK] Loaded vector store from: {persist_path}")
    
    return vector_store


def create_retriever(vector_store, k: int = 3):
    """创建检索器"""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever


def main():
    """主函数：演示向量数据库创建流程"""
    print("=" * 60)
    print("2.2 创建向量数据库")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("第一部分：演示踩坑 - 坑一")
    print("=" * 60)
    demo_pit1()
    
    print("\n" + "=" * 60)
    print("第二部分：演示踩坑 - 坑二")
    print("=" * 60)
    demo_pit2()
    
    sys.path.insert(0, os.path.dirname(__file__))
    
    try:
        from chapter_2_1_document_loading import load_document, split_document
        
        print("\n[Step 1] Load and split document...")
        doc_path = os.path.join(os.path.dirname(__file__), "docs", "example.txt")
        content = load_document(doc_path)
        chunks = split_document(content, chunk_size=300, chunk_overlap=30)
        print(f"Document split completed, {len(chunks)} chunks")
        
        print("\n[Step 2] Create embedding model...")
        embeddings = create_embeddings()
        
        print("\n[Step 3] Create vector store...")
        persist_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")
        print(f"Vector store path: {persist_path}")
        vector_store = create_vector_store(chunks, embeddings, persist_path)
        
        print("\n[Step 4] Create retriever and test...")
        retriever = create_retriever(vector_store, k=2)
        
        test_query = "什么是RAG？"
        print(f"\nTest query: {test_query}")
        retrieved_docs = retriever.invoke(test_query)
        print(f"Retrieved {len(retrieved_docs)} relevant documents")
        
        for i, doc in enumerate(retrieved_docs):
            print(f"\n--- Relevant Document {i+1} ---")
            print(f"Content: {doc.page_content[:150]}...")
            print(f"Metadata: {doc.metadata}")
            
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
