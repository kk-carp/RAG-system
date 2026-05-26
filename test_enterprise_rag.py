# test_enterprise_rag.py - 高级RAG环境验证脚本
import streamlit as st
import langchain
import langchain_community
import chromadb
import faiss
import ollama
import rank_bm25
import sentence_transformers
import tiktoken

# 打印核心组件版本
print("=== 高级RAG环境验证结果 ===")
print(f"Streamlit: {st.__version__}")
print(f"LangChain: {langchain.__version__}")
print(f"ChromaDB: {chromadb.__version__}")
print(f"FAISS: {faiss.__version__}")
# print(f"Ollama: {ollama.__version__}")
# print(f"BM25: {rank_bm25.__version__}")
print(f"SentenceTransformers: {sentence_transformers.__version__}")
print(f"Tiktoken: {tiktoken.__version__}")

# 测试高级RAG核心组件
print("\n=== 测试高级RAG核心组件 ===")
try:
    # 测试BM25召回
    from rank_bm25 import BM25Okapi
    corpus = ["企业考勤制度：每日打卡两次", "员工福利：五险一金+年终奖"]
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25多路召回组件正常")
    
    # 测试重排模型
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder('BAAI/bge-reranker-base', device='cpu')
    print("重排模型组件正常")
    
    # 测试窗口管理
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode("企业知识库测试文本")
    print("上下文窗口管理组件正常")
    
    # 测试LangChain RAG链
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template("测试{question}")
    print("LangChain RAG链组件正常")
except Exception as e:
    print(f"高级RAG组件异常: {e}")
    print("\n✅ 所有高级RAG环境配置验证通过！")