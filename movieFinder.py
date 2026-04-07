"""MovieFinder Agent 核心"""

import datetime
import os
import sys
import threading
from typing import Dict, List

from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langchain_classic.memory import ConversationSummaryMemory

from config import LLM_CONFIG, SYSTEM_PROMPT, RETRIEVAL_K, MEMORY_CONFIG
from apis.douban_tool import search_douban_movies

"""工具注册中心"""
TOOLS = [search_douban_movies]

class MovieFinder:
    def __init__(self):
        self.vector_store = None
        self.llm = None
        self.summary_llm = None
        self.tools = {t.name: t for t in TOOLS}
        self.memory = None
        self.history: List = []
        self.turn_counter = 0
        self.summary_lock = threading.Lock()
        self.log_dir = "logs"
        self._setup()
    
    def _setup(self):
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 配置 Loguru：只写入文件，不输出到控制台（避免重复）
        self._setup_logger()
        
        self._run_embedding_check()
        self._load_vector_store()
        self._setup_llm()
        self._setup_memory()
    
    def _setup_logger(self):
        """配置 Loguru 日志"""
        # 移除默认的 console 处理器
        logger.remove()
        
        # 创建会话日志文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"session_{timestamp}.log")
        
        # 添加文件处理器
        logger.add(
            log_file,
            format="{time:HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            encoding="utf-8",
            mode="a"  # 追加模式
        )
        
        # 可选：同时保留控制台输出（但用不同格式）
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level="WARNING",
            filter=lambda record: record["level"].name == "INFO"  # 控制台只显示 INFO 及以上
        )
        
        logger.info(f"=== MovieFinder 会话开始 ===")
        logger.info(f"日志文件: {log_file}")
        # print(f"✓ 日志系统初始化完成: {log_file}\n")
    
    def _format_message(self, msg) -> str:
        """格式化单条消息"""
        if isinstance(msg, tuple):
            role, content = msg
            return f"[{role.upper()}] {content[:200]}..."
        elif hasattr(msg, "type") and hasattr(msg, "content"):
            extra = ""
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                extra = f" | 工具调用: {msg.tool_calls}"
            if hasattr(msg, "tool_call_id"):
                extra = f" | 工具ID: {msg.tool_call_id}"
            return f"[{msg.type.upper()}] {msg.content[:200]}...{extra}"
        return str(msg)[:200]
    
    def _run_embedding_check(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        embed_dir = os.path.join(current_dir, "embed")
        start_py = os.path.join(embed_dir, "start.py")
        
        if os.path.exists(start_py):
            logger.debug("检查向量数据库...")
            sys.path.insert(0, embed_dir)
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("start", start_py)
                start_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(start_module)
                start_module.main()
                logger.debug("向量数据库检查完成")
            except Exception as e:
                logger.error(f"检查失败: {e}")
            finally:
                sys.path.pop(0)
    
    def _load_vector_store(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vector_store_path = os.path.join(current_dir, "rag", "vector_store")
        model_path = os.path.join(current_dir, "bge-small-zh-v1.5")
        
        if not os.path.exists(vector_store_path):
            logger.warning("向量数据库不存在")
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
            logger.info("向量库加载成功")
        except Exception as e:
            logger.error(f"向量库加载失败: {e}")
    
    def _setup_llm(self):
        config = LLM_CONFIG.copy()
        config["streaming"] = True
        self.llm = ChatOpenAI(**config).bind_tools(TOOLS)
        
        summary_config = LLM_CONFIG.copy()
        summary_config["streaming"] = False
        self.summary_llm = ChatOpenAI(**summary_config)
        logger.info("LLM 初始化完成")
    
    def _setup_memory(self):
        memory_config = MEMORY_CONFIG.copy()
        memory_config["llm"] = self.summary_llm
        self.memory = ConversationSummaryMemory(**memory_config)
        logger.info("ConversationSummaryMemory 初始化完成")
    
    def retrieve_context(self, query: str) -> str:
        if not self.vector_store:
            return ""
        try:
            results = self.vector_store.similarity_search(query, k=RETRIEVAL_K)
            contexts = []
            for i, doc in enumerate(results, 1):
                name = doc.metadata.get("movie_name", "未知")
                contexts.append(f"[{i}] 《{name}》: {doc.page_content[:100]}...")
            context_str = "\n".join(contexts)
            logger.debug(f"检索到上下文: {context_str[:200]}...")
            return context_str
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return f"检索失败: {e}"
    
    def execute_tool(self, tool_call: Dict) -> str:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        logger.info(f"调用工具: {tool_name}")
        logger.debug(f"工具参数: {tool_args}")
        
        if tool_name in self.tools:
            try:
                result = self.tools[tool_name].invoke(tool_args)
                result_preview = str(result)
                logger.info(f"工具返回: {result_preview}...")
                return result
            except Exception as e:
                err = f"工具执行失败: {str(e)}"
                logger.error(err)
                return err
        return f"未知工具: {tool_name}"
    
    def _background_summary(self, user_input: str, final_answer: str):
        try:
            with self.summary_lock:
                self.memory.save_context(
                    {"input": user_input},
                    {"output": final_answer}
                )
                logger.debug(f"[{threading.current_thread().name}] 后台总结完成")
        except Exception as e:
            logger.error(f"后台总结出错: {e}")
    
    def chat(self, user_input: str) -> str:
        self.turn_counter += 1
        logger.info(f"{'='*60}")
        logger.info(f"【第 {self.turn_counter} 轮对话】 用户输入: {user_input}")

        local_context = self.retrieve_context(user_input)
        summary = getattr(self.memory, "buffer", "")

        messages = [
            ("system", f"{SYSTEM_PROMPT}\n\n本地向量库资料:\n{local_context}\n\n历史对话总结：\n{summary}"),
        ]
        for msg in self.history[-8:]:      # 增加到最近8轮，提高记忆
            messages.append(msg)
        messages.append(("human", user_input))

        logger.info(f"发送给 LLM 的 Messages: {len(messages)} 条")

        # ==================== 流式输出核心 ====================
        print("\nAI: ", end="", flush=True)
        final_answer = ""

        # 先用 invoke 处理可能的工具调用（工具调用通常不需要流式）
        response = self.llm.invoke(messages)
        tool_call_count = 0

        while response.tool_calls:
            tool_call_count += 1
            logger.info(f">>> 第 {tool_call_count} 次工具调用")
            messages.append(response)

            for tool_call in response.tool_calls:
                result = self.execute_tool(tool_call)
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call.get("id", "")
                ))

            response = self.llm.invoke(messages)   # 继续直到没有工具调用

        # ==================== 最终回答真正流式输出 ====================
        # 强制走 stream，确保逐 token 输出
        stream_generator = self.llm.stream(messages)

        for chunk in stream_generator:
            if chunk.content:                     # 有些 chunk 可能为空
                print(chunk.content, end="", flush=True)
                final_answer += chunk.content

        print()   # 流式结束后换行

        logger.info(f"最终回答长度: {len(final_answer)} 字符")
        logger.info(f"第 {self.turn_counter} 轮结束 {'='*60}")

        # 更新历史
        self.history.append(("human", user_input))
        self.history.append(("ai", final_answer))
        if len(self.history) > 8:
            self.history = self.history[-8:]

        # 后台总结（保留）
        if self.turn_counter % 2 == 0:
            thread = threading.Thread(
                target=self._background_summary,
                args=(user_input, final_answer),
                daemon=True,
                name=f"Summary-Thread-{self.turn_counter}"
            )
            thread.start()
            logger.info(f"[{thread.name}] 后台总结线程已启动")

        return final_answer