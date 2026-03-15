"""MovieFinder Agent 核心"""

import os
import sys
from typing import Dict, List

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import LLM_CONFIG, SYSTEM_PROMPT, RETRIEVAL_K
from apis.douban_tool import search_douban_movies

"""工具注册中心"""
# 导出所有可用工具
TOOLS = [search_douban_movies]

class MovieFinder:
    def __init__(self):
        self.vector_store = None
        self.llm = None
        self.tools = {t.name: t for t in TOOLS}
        self.history: List = []
        self._setup()
    
    def _setup(self):
        self._run_embedding_check()
        self._load_vector_store()
        self._setup_llm()
    
    def _run_embedding_check(self):
        """检查 embedding"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        embed_dir = os.path.join(current_dir, "embed")
        start_py = os.path.join(embed_dir, "start.py")
        
        if os.path.exists(start_py):
            print("检查向量数据库...")
            sys.path.insert(0, embed_dir)
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("start", start_py)
                start_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(start_module)
                start_module.main()
                print("检查完成\n")
            except Exception as e:
                print(f"检查失败: {e}")
            finally:
                sys.path.pop(0)
    
    def _load_vector_store(self):
        """加载向量库"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vector_store_path = os.path.join(current_dir, "rag", "vector_store")
        model_path = os.path.join(current_dir, "bge-small-zh-v1.5")
        
        if not os.path.exists(vector_store_path):
            print("向量数据库不存在\n")
            return
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={'device': 'cpu', 'local_files_only': True},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.vector_store = FAISS.load_local(
                vector_store_path, embeddings, 
                allow_dangerous_deserialization=True
            )
            print("向量库加载成功\n")
        except Exception as e:
            print(f"向量库加载失败: {e}\n")
    
    def _setup_llm(self):
        """初始化 LLM"""
        self.llm = ChatOpenAI(**LLM_CONFIG).bind_tools(TOOLS)
    
    def retrieve_context(self, query: str) -> str:
        """检索本地向量库"""
        if not self.vector_store:
            return ""
        try:
            results = self.vector_store.similarity_search(query, k=RETRIEVAL_K)
            contexts = []
            for i, doc in enumerate(results, 1):
                name = doc.metadata.get("movie_name", "未知")
                contexts.append(f"[{i}] 《{name}》: {doc.page_content[:100]}...")
            return "\n".join(contexts)
        except Exception as e:
            return f"检索失败: {e}"
    
    def execute_tool(self, tool_call: Dict) -> str:
        """执行工具调用"""
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        print(f"\n>>> 调用工具: {tool_name}")
        print(f"参数: {tool_args}")
        
        if tool_name in self.tools:
            try:
                result = self.tools[tool_name].invoke(tool_args)
                print(f"<<< 工具返回: {result[:200]}...")
                return result
            except Exception as e:
                return f"工具执行失败: {str(e)}"
        return f"未知工具: {tool_name}"
    
    def chat(self, user_input: str) -> str:
        """Agent 对话流程"""
        from langchain_core.messages import ToolMessage
        
        local_context = self.retrieve_context(user_input)
        
        # 构建消息
        messages = [
            ("system", f"{SYSTEM_PROMPT}\n\n本地向量库资料:\n{local_context}"),
            *self.history[-6:],
            ("human", user_input)
        ]
        
        print("=" * 60)
        print("【发送给 AI】")
        print(f"用户: {user_input}")
        print(f"本地上下文: {local_context[:200]}...")
        print("=" * 60)
        
        # 调用 LLM
        response = self.llm.invoke(messages)
        
        # 处理工具调用
        while response.tool_calls:
            print(f"\n>>> 检测到 {len(response.tool_calls)} 个工具调用")
            
            # 添加 AI 的 tool_calls 到历史
            messages.append(response)
            
            # 执行每个工具
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]  # 关键：获取调用ID
                
                print(f">>> 执行工具: {tool_name} | ID: {tool_call_id}")
                
                if tool_name in self.tools:
                    try:
                        result = self.tools[tool_name].invoke(tool_args)
                        print(f"<<< 工具返回: {str(result)[:200]}...")
                    except Exception as e:
                        result = f"工具执行失败: {str(e)}"
                        print(f"<<< 工具错误: {result}")
                else:
                    result = f"未知工具: {tool_name}"
                    print(f"<<< 未知工具: {tool_name}")
                
                # 关键：添加 ToolMessage 时必须包含 tool_call_id
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id  # 必须匹配对应的 tool_call
                ))
            
            print("\n>>> AI 继续生成...")
            response = self.llm.invoke(messages)
        
        # 最终回答
        final_answer = response.content or ""
        
        # 更新历史（简化，只保留文本消息）
        self.history.append(("human", user_input))
        self.history.append(("ai", final_answer))
        
        if len(self.history) > 20:
            self.history = self.history[-20:]
        
        return final_answer