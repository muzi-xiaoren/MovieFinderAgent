import os
import sys
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import LLM_CONFIG, SYSTEM_PROMPT, SESSION_ID, RETRIEVAL_K


class MovieFinder:
    def __init__(self):
        self.store: Dict[str, BaseChatMessageHistory] = {}
        self.vector_store = None
        self.chain_with_history = None
        self._setup()
    
    def _setup(self):
        """初始化配置"""
        self._run_embedding_check()
        self._load_vector_store()
        self._setup_chain()
    
    def _run_embedding_check(self):
        """调用 embed/start.py 进行 embedding 检查"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        rag_dir = os.path.join(current_dir, "rag")
        start_py = os.path.join(rag_dir, "start.py")
        
        if os.path.exists(start_py):
            print("正在检查向量数据库...")
            sys.path.insert(0, rag_dir)
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("start", start_py)
                start_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(start_module)
                start_module.main()
                print("向量数据库检查完成\n")
            except Exception as e:
                print(f"embedding 检查失败: {e}")
            finally:
                sys.path.pop(0)
        else:
            print(f"警告: 未找到 {start_py}")
    
    def _load_vector_store(self):
        """加载 FAISS 向量数据库"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vector_store_path = os.path.join(current_dir, "embed", "vector_store")
        model_path = os.path.join(current_dir, "bge-small-zh-v1.5")
        
        if not os.path.exists(vector_store_path):
            print(f"错误: 向量数据库不存在: {vector_store_path}")
            return
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={'device': 'cpu', 'local_files_only': True},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.vector_store = FAISS.load_local(
                vector_store_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print(f"成功加载向量数据库\n")
        except Exception as e:
            print(f"加载向量数据库失败: {e}")
    
    def _setup_chain(self):
        """设置 LLM 链"""
        llm = ChatOpenAI(**LLM_CONFIG)
        parser = StrOutputParser()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        
        chain = prompt | llm | parser
        
        self.chain_with_history = RunnableWithMessageHistory(
            chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
    
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """获取会话历史"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def retrieve_context(self, query: str) -> str:
        """从向量数据库检索相关电影信息"""
        if self.vector_store is None:
            return "（向量数据库未加载）"
        
        try:
            results = self.vector_store.similarity_search(query, k=RETRIEVAL_K)
            contexts = []
            for i, doc in enumerate(results, 1):
                movie_name = doc.metadata.get("movie_name", "未知")
                contexts.append(f"[{i}] 《{movie_name}》: {doc.page_content}")
            return "\n\n".join(contexts)
        except Exception as e:
            return f"（检索失败: {e}）"
    
    def get_full_input(self, user_input: str) -> Dict[str, Any]:
        """构建发送给 AI 的完整输入"""
        context = self.retrieve_context(user_input)
        return {
            "input": user_input,
            "context": context
        }
    
    def stream_response(self, full_input: Dict[str, Any]):
        """流式获取 AI 响应"""
        return self.chain_with_history.stream(
            full_input,
            config={"configurable": {"session_id": SESSION_ID}},
        )