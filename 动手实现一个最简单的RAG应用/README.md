# 动手实现一个最简单的RAG应用

本项目根据飞书教程《【AI大模型应用开发】3.RAG初探 - 动手实现一个最简单的RAG应用》实现，包含完整的RAG应用代码，每个模块对应教程的一个章节。

## 项目结构

```
动手实现一个最简单的RAG应用/
├── 0_什么是RAG.md                      # RAG概念介绍
├── 1_RAG基本流程.md                    # RAG流程概述
├── chapter_2_1_document_loading.py     # 文档加载与分块
├── chapter_2_2_vector_store.py         # 创建向量数据库（含踩坑说明）
├── chapter_3_prompt_template.py        # Prompt模板设计
├── chapter_4_llm_answer.py             # 使用大模型生成答案
├── chapter_5_rag_wrapper.py            # 完整RAG封装代码
├── README.md                           # 项目说明
├── docs/
│   └── example.txt                     # 示例文档
└── vector_store/                       # 向量数据库（运行后自动生成）
```

## 安装依赖

```bash
# 使用项目虚拟环境
cd C:\Users\Administrator\Desktop\PythonProject
.\.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 配置环境变量

确保项目根目录下的`.env`文件已正确配置：

```env
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=你的API基础地址
```

## 使用方法

### 1. 分步运行（学习模式）

按照教程顺序依次运行每个模块：

```bash
# 步骤1：文档加载与分块
python chapter_2_1_document_loading.py

# 步骤2：创建向量数据库
python chapter_2_2_vector_store.py

# 步骤3：Prompt模板
python chapter_3_prompt_template.py

# 步骤4：使用大模型得到答案
python chapter_4_llm_answer.py
```

### 2. 完整运行（封装模式）

直接运行封装好的RAG类：

```bash
python chapter_5_rag_wrapper.py
```

### 3. 编程接口

```python
from chapter_5_rag_wrapper import SimpleRAG

# 创建RAG实例
rag = SimpleRAG()

# 添加文档
rag.add_document("docs/example.txt")

# 查询
result = rag.query("什么是RAG？")
print(result["answer"])
```

## 教程章节对应关系

| 教程章节 | 文件 | 内容 |
|----------|------|------|
| 0. 什么是RAG | 0_什么是RAG.md | RAG概念介绍 |
| 1. RAG基本流程 | 1_RAG基本流程.md | RAG流程概述 |
| 2.1 文档加载与分块 | chapter_2_1_document_loading.py | 文档加载与分块实现 |
| 2.2 创建向量数据库 | chapter_2_2_vector_store.py | 向量数据库创建 |
| 2.2.3 踩坑 | chapter_2_2_vector_store.py | 包含坑一和坑二的处理 |
| 3. Prompt模板 | chapter_3_prompt_template.py | Prompt模板设计 |
| 4.1 封装API | chapter_4_llm_answer.py | 大模型API封装 |
| 4.2 组装Prompt | chapter_4_llm_answer.py | Prompt组装 |
| 4.3 使用大模型 | chapter_4_llm_answer.py | 生成答案 |
| 5.1 封装RAG | chapter_5_rag_wrapper.py | RAG类封装 |
| 5.2 完整代码 | chapter_5_rag_wrapper.py | 完整代码演示 |

## 常见问题

### 问题1：NoneType object is not iterable

**原因**：API Key或Base URL配置错误导致模型初始化失败

**解决**：检查`.env`文件中的`OPENAI_API_KEY`和`OPENAI_BASE_URL`是否正确设置

### 问题2：Number of embeddings 9 must match number of ids 10

**原因**：文档列表中包含空文档或None

**解决**：代码中已添加空文档过滤逻辑，确保每个文档块都有内容

### 问题3：ModuleNotFoundError

**原因**：缺少相关依赖包

**解决**：运行 `pip install -r requirements.txt` 安装所有依赖

## 技术栈

- Python 3.10+
- LangChain Core
- LangChain OpenAI
- LangChain Community
- LangChain Text Splitters
- FAISS（向量数据库）
- python-dotenv
