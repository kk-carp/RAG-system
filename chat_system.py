import os
# 配置Hugging Face国内镜像，解决无法连接官方源的问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import re
import time
import logging
import streamlit as st
import ollama
from dotenv import load_dotenv
from typing import List, Dict, Any

# LangChain核心组件（仅保留必要模块，减少导入耗时）
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, DirectoryLoader,
    Docx2txtLoader, UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import jieba

# 多路召回+重排相关（仅保留轻量实现）
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# 基础优化：禁用冗余日志，减少IO开销
logging.basicConfig(level=logging.ERROR)
st.set_page_config(page_title="工业级企业知识库问答系统", page_icon="🏭", layout="wide")



# 加载环境变量
load_dotenv()

# ======================== 全局配置（极致速度优化） ========================
LLM_MODEL = "deepseek-r1:1.5b"
# 替换为轻量嵌入模型（速度提升5倍）
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# 轻量重排模型（base版比large快3倍）
RERANK_MODEL = "BAAI/bge-reranker-base"

# 核心优化：最小化计算量
CONTEXT_WINDOW_SIZE = 1024  # 进一步缩小窗口，减少token计算
CHUNK_SIZE = 300  # 适度增大分块，减少分块总数（中文≈200字）
CHUNK_OVERLAP = 30  # 10%重叠率，平衡语义和冗余
TOP_K_RECALL = 8  # 召回数量减半，减少重排计算
TOP_K_RERANK = 3  # 仅保留前3个最相关文档
MAX_DISPLAY_CHARS = 400  # 文档块内容预览长度

# 全局缓存：预计算的BM25索引
bm25_index_cache = None


# ======================== 通用文档加载器（优化IO） ========================
def load_single_file(file_path: str) -> List[Document]:
    """加载单个文件（简化异常处理，减少耗时）"""
    docs = []
    try:
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.lower().endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            return []  # 跳过不支持的格式，减少处理时间

        docs = loader.load()
        st.success(f"✅ 成功加载单个文件：{os.path.basename(file_path)}")
    except Exception as e:
        st.error(f"❌ 加载失败：{str(e)[:50]}")  # 缩短错误信息，减少IO
    return docs


def load_directory(dir_path: str) -> List[Document]:
    """加载文件夹（拆分异常处理，避免单文件损坏导致整体中断）"""
    docs = []
    
    # 加载 PDF 文件
    try:
        pdf_loader = DirectoryLoader(
            dir_path, glob="*.pdf", loader_cls=PyPDFLoader, show_progress=False
        )
        docs.extend(pdf_loader.load())
    except Exception as e:
        st.warning(f"⚠️ 加载PDF文件出错：{str(e)[:50]}")

    # 加载 TXT 文件
    try:
        text_loader = DirectoryLoader(
            dir_path, glob="*.txt", loader_cls=TextLoader, show_progress=False,
            loader_kwargs={"encoding": "utf-8"}
        )
        docs.extend(text_loader.load())
    except Exception as e:
        st.warning(f"⚠️ 加载TXT文件出错：{str(e)[:50]}")

    # 加载 DOCX 文件（常见 File is not a zip file 错误隔离）
    try:
        docx_loader = DirectoryLoader(
            dir_path, glob="*.docx", loader_cls=Docx2txtLoader, show_progress=False
        )
        docs.extend(docx_loader.load())
    except Exception as e:
        st.warning(f"⚠️ 加载DOCX文件出错：{str(e)[:50]}（可能存在损坏或非标准docx文件）")

    if docs:
        st.success(f"✅ 加载 {len(docs)} 个文档")
    else:
        st.error("❌ 加载文件夹失败：未成功加载任何文档")
        
    return docs



def load_documents(doc_path: str = "./docs") -> List[Document]:
    """通用加载入口（简化路径检查）"""
    doc_path = os.path.abspath(doc_path)
    if not os.path.exists(doc_path):
        st.error(f"❌ 路径不存在：{doc_path}")
        return []
    return load_single_file(doc_path) if os.path.isfile(doc_path) else load_directory(doc_path)


# ======================== 文档处理（预计算+轻量嵌入） ========================
def split_documents(docs: List[Document]) -> List[Document]:
    """文档分块（简化分隔符，提升速度）"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？"],  # 仅保留中文核心分隔符
        length_function=len
    )
    splits = text_splitter.split_documents(docs)
    st.info(f"📄 生成 {len(splits)} 个文本块")
    return splits


def init_bm25_index(splits: List[Document]):
    """预计算BM25索引，仅初始化一次"""
    global bm25_index_cache
    if bm25_index_cache is None and splits:
        texts = [doc.page_content for doc in splits]
        tokenized_texts = [jieba.lcut(text) for text in texts]
        bm25_index_cache = BM25Okapi(tokenized_texts)


def build_vector_db(splits: List[Document], persist_dir: str = "./chroma_db") -> Chroma:
    """构建向量库（使用轻量本地嵌入模型，替代Ollama）"""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # 轻量嵌入模型配置（多线程编码）
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},  # 批量编码提速
        multi_process=True  # 多进程加速
    )

    if os.path.exists(persist_dir):
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        st.info("♻️ 加载已有向量库")
    else:
        # 批量插入，减少IO次数
        vectordb = Chroma.from_documents(
            documents=splits, embedding=embeddings, persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"}  # 优化检索算法
        )
        vectordb.persist()
        st.success("✅ 向量库构建完成")

    return vectordb


# ======================== 多路召回（极致优化） ========================
def bm25_retriever(query: str, splits: List[Document]) -> List[Document]:
    """BM25召回（使用预计算索引，无冗余计算）"""
    if bm25_index_cache is None or not splits:
        return []

    tokenized_query = jieba.lcut(query)
    scores = bm25_index_cache.get_scores(tokenized_query)
    top_n_indices = scores.argsort()[-TOP_K_RECALL:][::-1]

    bm25_docs = []
    for idx in top_n_indices:
        doc = splits[idx]
        doc.metadata["retrieval_type"] = "bm25"
        doc.metadata["score"] = float(scores[idx])
        bm25_docs.append(doc)
    return bm25_docs


# ======================== 修复后的 vector_retriever ========================
def vector_retriever(query: str, vectordb: Chroma, top_k: int = TOP_K_RECALL) -> List[Document]:
    """向量语义召回（增加分数处理，确保返回不为空）"""
    try:
        vector_docs = vectordb.similarity_search_with_score(query, k=top_k)
        result_docs = []
        for doc, score in vector_docs:
            # 确保创建新的文档对象，避免修改原对象引用
            new_doc = Document(
                page_content=doc.page_content,
                metadata=doc.metadata.copy()
            )
            new_doc.metadata["retrieval_type"] = "vector"
            new_doc.metadata["score"] = float(score) if score is not None else 0.0
            result_docs.append(new_doc)
        return result_docs
    except Exception as e:
        st.warning(f"向量检索失败: {str(e)}")
        return []


# ======================== 修复后的 rerank_documents ========================
def rerank_documents(query: str, docs: List[Document], top_k: int = TOP_K_RERANK) -> List[Document]:
    """重排模型（增加兜底保护，避免空列表）"""
    if not docs:
        st.warning("无相关文档可重排")
        return []

    try:
        # 尝试加载轻量版重排模型
        reranker = CrossEncoder('BAAI/bge-reranker-base', device='cpu')
        # 限制最大长度，避免OOM
        pairs = [[query, doc.page_content[:200]] for doc in docs]
        scores = reranker.predict(pairs)

        doc_score_pairs = []
        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = float(score)
            doc_score_pairs.append((doc, score))

        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in doc_score_pairs[:top_k]]
        return reranked_docs
    except Exception as e:
        st.warning(f"重排失败，使用向量分数: {str(e)}")
        # 兜底：直接返回前top_k，不进行重排
        return docs[:top_k] if len(docs) > top_k else docs

def window_management(docs: List[Document], max_tokens: int = CONTEXT_WINDOW_SIZE) -> List[Document]:
    """上下文窗口管理（适配 2048 窗口大小）"""
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    selected_docs = []

    for doc in docs:
        doc_tokens = len(encoder.encode(doc.page_content))
        if total_tokens + doc_tokens <= max_tokens:
            selected_docs.append(doc)
            total_tokens += doc_tokens
        else:
            # 截断超出窗口的部分
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 0:
                truncated_content = encoder.decode(encoder.encode(doc.page_content)[:remaining_tokens])
                doc.page_content = truncated_content
                selected_docs.append(doc)
            break

    return selected_docs
# ======================== 修复后的 multi_retrieval_pipeline ========================
def multi_retrieval_pipeline(query: str, vectordb: Chroma, splits: List[Document]) -> List[Document]:
    """多路召回流水线（修复去重逻辑，保留所有文档）"""
    self_query_docs = []  # 已移除Self-query，保持空列表
    bm25_docs = bm25_retriever(query, splits)
    vector_docs = vector_retriever(query, vectordb)

    # 🔴 修复核心：使用内容指纹+来源去重，避免错误过滤
    all_docs = self_query_docs + bm25_docs + vector_docs
    unique_docs = []
    seen_signatures = set()

    for doc in all_docs:
        # 生成唯一签名：来源 + 前50字符（更稳定）
        if doc.page_content:
            content_preview = doc.page_content[:50].strip()
            source = doc.metadata.get('source', 'unknown')
            signature = f"{source}_{content_preview}"

            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_docs.append(doc)

                # 提前停止，避免处理过多文档
                if len(unique_docs) >= TOP_K_RECALL:
                    break

    # 重排
    reranked_docs = rerank_documents(query, unique_docs)

    # 窗口管理
    final_docs = window_management(reranked_docs)

    # 按得分排序，优先使用重排得分；若无则使用原始检索得分
    final_docs.sort(
        key=lambda doc: float(doc.metadata.get("rerank_score", doc.metadata.get("score", 0.0))),
        reverse=True
    )

    # 存储本次检索结果，方便界面展示
    st.session_state["retrieved_docs"] = final_docs

    # 调试：打印找到的文档数量
    st.sidebar.caption(f"检索到文档数: {len(final_docs)}")
    return final_docs


# ======================== 构建RAG链（简化提示词） ========================
def build_rag_chain(retrieved_docs: List[Document]):
    """构建RAG链（修复Prompt类型错误）"""
    # 简化提示词，适配小模型
    prompt = ChatPromptTemplate.from_template("""
    你是企业知识库问答助手，严格基于上下文回答问题，不要编造信息。
    如果没有相关信息，直接说"未找到相关答案"。
    回答要简洁、准确，控制在100字以内。

    上下文：
    {context}

    问题：{question}

    回答：
    """)

    def format_docs(docs):
        return "\n\n".join([f"文档 {i + 1}：{doc.page_content}" for i, doc in enumerate(docs)])

    # 核心修复：自定义函数将Prompt对象转为纯字符串
    def convert_prompt_to_string(prompt_value):
        """将ChatPromptValue转换为纯字符串"""
        # 提取Prompt中的文本内容（兼容新版LangChain）
        if hasattr(prompt_value, 'to_string'):
            return prompt_value.to_string()
        # 兜底：拼接messages中的内容
        elif hasattr(prompt_value, 'messages'):
            return "\n".join([msg.content for msg in prompt_value.messages])
        else:
            return str(prompt_value)

    rag_chain = (
            {"context": lambda x: format_docs(retrieved_docs),
             "question": RunnablePassthrough()}
            | prompt
            # 第一步：转换Prompt为纯字符串
            | RunnableLambda(lambda x: convert_prompt_to_string(x))
            # 第二步：传给Ollama（此时已是纯字符串）
            | RunnableLambda(lambda x: ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": x}]))
            | RunnableLambda(lambda x: x["message"]["content"])
            | StrOutputParser()
    )

    return rag_chain
# ======================== Streamlit界面（简化交互） ========================
def main():
    st.title("🏭 企业知识库问答系统（极速版）")

    with st.sidebar:
        st.header("📚 知识库管理")
        doc_dir = st.text_input("文档路径", value="./docs")
        if st.button("🔄 加载并构建", type="primary"):
            with st.spinner("加载中..."):
                docs = load_documents(doc_dir)
            if docs:
                with st.spinner("分块中..."):
                    splits = split_documents(docs)
                with st.spinner("初始化索引..."):
                    init_bm25_index(splits)  # 预计算BM25
                with st.spinner("构建向量库..."):
                    vectordb = build_vector_db(splits)
                st.session_state["vectordb"] = vectordb
                st.session_state["splits"] = splits
                st.success("✅ 知识库就绪！")

    # 问答区域
    user_query = st.text_input("请输入你的问题", placeholder="员工准则是什么？")
    if st.button("🚀 提交", type="primary") and user_query:
        if "vectordb" not in st.session_state:
            st.error("请先加载知识库！")
            return

        with st.spinner("检索文档..."):
            retrieved_docs = multi_retrieval_pipeline(user_query, st.session_state["vectordb"], st.session_state["splits"])
            st.session_state["retrieved_docs"] = retrieved_docs

        with st.spinner("思考中..."):
            start_time = time.time()
            rag_chain = build_rag_chain(retrieved_docs)
            answer = rag_chain.invoke({"question": user_query})
            total_time = time.time() - start_time

        st.subheader("🎯 回答结果")
        st.write(answer)
        st.caption(f"总耗时：{total_time:.2f}秒（目标≤2秒）")

        if st.session_state["retrieved_docs"]:
            with st.expander("📄 检索到的文档块"):
                for idx, doc in enumerate(st.session_state["retrieved_docs"], start=1):
                    source = doc.metadata.get("source", "未知")
                    retrieval_type = doc.metadata.get("retrieval_type", "未知")
                    display_score = float(doc.metadata.get("rerank_score", doc.metadata.get("score", 0.0)))
                    score_label = "重排得分" if "rerank_score" in doc.metadata else "检索得分"
                    st.markdown(f"**{idx}. 来源：{source} | 方式：{retrieval_type} | {score_label}：{display_score:.4f}**")
                    preview = doc.page_content[:MAX_DISPLAY_CHARS]
                    if len(doc.page_content) > MAX_DISPLAY_CHARS:
                        preview += "..."
                    st.write(preview)
                    if len(doc.page_content) > MAX_DISPLAY_CHARS:
                        with st.expander("查看完整文档块"):
                            st.write(doc.page_content)
                    st.markdown("---")

if __name__ == "__main__":
    main()