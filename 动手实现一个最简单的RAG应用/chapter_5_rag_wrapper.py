#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5. 封装RAG

本模块实现RAG流程的完整封装，将所有步骤整合到一个类中，提供简洁的API接口。

教程目录对应：
- 5.1 封装RAG
- 5.2 完整代码

主要功能：
1. 将文档加载、分块、向量化、检索、生成等步骤封装为一个类
2. 提供简洁的查询接口
3. 支持增量添加文档
4. 支持向量数据库的持久化和加载
5. 提供详细的日志输出和错误处理
"""

import os
import sys
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
_ = load_dotenv(find_dotenv())

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.documents import Document
except ImportError as e:
    print(f"❌ 导入依赖失败: {e}")
    print("💡 请安装所需依赖: pip install langchain-openai langchain-community langchain-text-splitters")
    sys.exit(1)


class SimpleRAG:
    """
    5.1 封装RAG
    
    一个简单的RAG应用封装类，整合了文档加载、分块、向量化、检索和生成的完整流程。
    
    使用方法：
        # 创建RAG实例
        rag = SimpleRAG()
        
        # 添加文档
        rag.add_document("docs/example.txt")
        
        # 查询
        result = rag.query("什么是RAG？")
        
        # 打印结果
        print(result["answer"])
    """
    
    def __init__(
        self,
        persist_path: str = None,
        model_name: str = "deepseek-chat",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 300,
        chunk_overlap: int = 30,
        k: int = 3,
        temperature: float = 0.3,
    ):
        """
        初始化RAG实例
        
        Args:
            persist_path: 向量数据库持久化路径
            model_name: 大模型名称
            embedding_model: 嵌入模型名称
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠大小
            k: 检索返回的文档数量
            temperature: 大模型温度参数
        """
        # 重要：FAISS的C++底层无法处理中文路径，必须使用纯ASCII路径
        # 默认将向量数据库保存到项目根目录（不包含中文）
        if persist_path is None:
            persist_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")
        self.persist_path = persist_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.temperature = temperature
        
        # 初始化组件
        self._init_embeddings(embedding_model)
        self._init_llm(model_name)
        self._init_prompt_template()
        self._init_vector_store()
        
        print("✅ SimpleRAG 初始化完成")
    
    def _init_embeddings(self, model_name: str):
        """初始化嵌入模型"""
        try:
            self.embeddings = OpenAIEmbeddings(
                model=model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
            print(f"   - 嵌入模型: {model_name}")
        except Exception as e:
            print(f"❌ 初始化嵌入模型失败: {e}")
            raise
    
    def _init_llm(self, model_name: str):
        """初始化大语言模型"""
        try:
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                temperature=self.temperature,
                timeout=30,
                max_retries=2,
            )
            self.output_parser = StrOutputParser()
            print(f"   - 大模型: {model_name}")
        except Exception as e:
            print(f"❌ 初始化大模型失败: {e}")
            raise
    
    def _init_prompt_template(self):
        """初始化Prompt模板"""
        template = """
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
"""
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )
        print("   - Prompt模板: RAG专用模板")
    
    def _init_vector_store(self):
        """初始化向量数据库"""
        if os.path.exists(self.persist_path):
            # 加载已存在的向量数据库
            try:
                self.vector_store = FAISS.load_local(
                    self.persist_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"   - 向量数据库: 从 {self.persist_path} 加载")
            except Exception as e:
                print(f"⚠️ 加载向量数据库失败，将创建新的: {e}")
                self.vector_store = None
        else:
            # 创建新的向量数据库
            self.vector_store = None
            print("   - 向量数据库: 待创建")
    
    def _split_document(self, content: str) -> list:
        """将文档内容分块"""
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n",
            length_function=len,
        )
        chunks = text_splitter.create_documents([content])
        
        # 添加元数据
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
    
    def add_document(self, file_path: str):
        """
        添加文档到RAG系统
        
        Args:
            file_path: 文档文件路径
            
        流程：
            1. 读取文档内容
            2. 分块处理
            3. 向量化并添加到向量数据库
            4. 持久化保存
        """
        print(f"\n📄 添加文档: {file_path}")
        
        # 读取文档
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read()
        
        print(f"   - 文档大小: {len(content)} 字符")
        
        # 分块
        chunks = self._split_document(content)
        print(f"   - 分块数量: {len(chunks)}")
        
        # 添加到向量数据库
        if self.vector_store is None:
            # 创建新的向量数据库
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            # 添加到现有向量数据库
            self.vector_store.add_documents(chunks)
        
        # 确保存储目录存在
        os.makedirs(self.persist_path, exist_ok=True)
        
        # 持久化
        self.vector_store.save_local(self.persist_path)
        print(f"✅ 文档已添加并保存到: {self.persist_path}")
    
    def add_documents(self, file_paths: list):
        """批量添加文档"""
        for file_path in file_paths:
            self.add_document(file_path)
    
    def query(self, question: str, k: int = None) -> dict:
        """
        执行RAG查询
        
        Args:
            question: 用户问题
            k: 返回的相关文档数量（默认为初始化时设置的值）
            
        Returns:
            包含问题、回答、检索到的文档和Prompt的字典
            
        流程：
            1. 检查向量数据库是否存在
            2. 使用检索器获取相关文档
            3. 构建Prompt
            4. 调用大模型生成回答
            5. 返回结果
        """
        if k is None:
            k = self.k
        
        print(f"\n🔍 查询: {question}")
        
        # 检查向量数据库
        if self.vector_store is None:
            raise ValueError("向量数据库为空，请先添加文档")
        
        # 创建检索器
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
        
        # 步骤1：检索相关文档
        print(f"   [1/3] 检索相关文档...")
        retrieved_docs = retriever.invoke(question)
        print(f"         检索到 {len(retrieved_docs)} 个相关文档")
        
        # 步骤2：构建Prompt
        print(f"   [2/3] 构建Prompt...")
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            content = doc.page_content.strip()
            if content:
                context_parts.append(f"【文档{i+1}】\n{content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        if not context:
            context = "（未检索到相关文档）"
        
        prompt = self.prompt_template.format(
            context=context,
            question=question,
        )
        
        # 步骤3：调用大模型生成回答
        print(f"   [3/3] 生成回答...")
        response = self.llm.invoke(prompt)
        answer = self.output_parser.invoke(response)
        
        print(f"✅ 查询完成")
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "prompt": prompt,
            "num_retrieved": len(retrieved_docs),
        }


def main():
    """
    5.2 完整代码
    
    主函数：演示SimpleRAG类的完整使用流程
    """
    print("=" * 60)
    print("5. 封装RAG - 完整代码演示")
    print("=" * 60)
    
    try:
        # 步骤1：创建RAG实例
        print("\n[步骤1] 创建SimpleRAG实例")
        print("-" * 40)
        rag = SimpleRAG(
            persist_path=os.path.join(os.path.dirname(__file__), "vector_store"),
            model_name="deepseek-chat",
            embedding_model="text-embedding-3-small",
            chunk_size=300,
            chunk_overlap=30,
            k=3,
            temperature=0.3,
        )
        
        # 步骤2：添加文档（如果向量数据库为空）
        print("\n[步骤2] 添加文档")
        print("-" * 40)
        doc_path = os.path.join(os.path.dirname(__file__), "docs", "example.txt")
        rag.add_document(doc_path)
        
        # 步骤3：执行查询
        print("\n[步骤3] 执行查询")
        print("-" * 40)
        
        # 查询1：什么是RAG
        print("\n查询1: 什么是RAG？")
        result1 = rag.query("什么是RAG？")
        print(f"\n回答:\n{result1['answer']}")
        
        # 查询2：RAG的核心流程
        print("\n\n查询2: RAG的核心流程包括哪些步骤？")
        result2 = rag.query("RAG的核心流程包括哪些步骤？")
        print(f"\n回答:\n{result2['answer']}")
        
        # 查询3：文档分块的重要性
        print("\n\n查询3: 文档分块在RAG中有什么重要性？")
        result3 = rag.query("文档分块在RAG中有什么重要性？")
        print(f"\n回答:\n{result3['answer']}")
        
        # 查询4：向量数据库的作用
        print("\n\n查询4: 向量数据库在RAG中起到什么作用？")
        result4 = rag.query("向量数据库在RAG中起到什么作用？")
        print(f"\n回答:\n{result4['answer']}")
        
        print("\n" + "=" * 60)
        print("🎉 RAG应用演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 出错了: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
