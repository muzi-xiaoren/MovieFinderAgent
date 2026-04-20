# streamlit run app.py
import streamlit as st
import sys
from pathlib import Path
import random
from movieFinder import MovieFinder
import asyncio

# Windows 兼容性修复
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ====================== 添加项目路径 ======================
BASE_DIR = Path(__file__).parent.absolute()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

st.set_page_config(
    page_title="电影摸鱼王 Agent",
    page_icon="🍿",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🍿 电影摸鱼王 Agent 🎬")
st.markdown("**随便问我电影的事，我去豆瓣 + RAG 帮你挖得干干净净！** 😎")

# ====================== 侧边栏 ======================
with st.sidebar:
    st.header("🎥 关于我")
    st.markdown("""
    - **核心**：LangChain + 豆瓣工具 + RAG  
    - **特点**：支持多轮对话、工具调用、向量检索  
    - **模型**：使用你 config.py 中配置的 LLM
    """)
    
    st.divider()
    
    if st.button("🎲 来一个电影烂梗"):
        jokes = [
            "为什么《阿凡达2》那么长？因为导演想让你在电影院多摸会儿鱼～",
            "看电影不哭的都是狠人……除非是喜剧把我笑哭了",
            "我最爱的电影类型：下班后能让我立刻睡着的",
            "《盗梦空间》告诉我们：多层梦境也逃不过加班"
        ]
        st.info(random.choice(jokes))
    
    st.caption("当前后端：MovieFinder（chat 方法 + 流式输出）")

# ====================== 初始化 Agent ======================
if "agent" not in st.session_state:
    with st.spinner("正在初始化电影摸鱼王 Agent..."):
        try:
            st.session_state.agent = MovieFinder()
            st.success("✅ Agent 初始化完成！")
        except Exception as e:
            st.error(f"初始化失败: {e}")
            st.stop()

# 初始化欢迎消息
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "嘿！我是**电影摸鱼王** 👑\n\n"
                       "随便问我电影的事：\n"
                       "• 最近好看的科幻片\n"
                       "• 诺兰电影按评分排序\n"
                       "• 《寄生虫》类似推荐\n"
                       "• 演员阵容、豆瓣评分...\n\n"
                       "来吧，甩问题过来！"
        }
    ]

# ====================== 显示聊天历史 ======================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 防止 ghosting 的 hack
st.empty()

# ====================== 用户输入处理 ======================
if prompt := st.chat_input("输入你的电影问题，例如：最近好看的悬疑片？"):
    
    # 添加并显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ==================== Assistant 回复（核心修复区域） ====================
    with st.chat_message("assistant"):
        # 先放一个 empty 减少 ghosting
        st.empty()
        
        # 思考提示
        thinking_placeholder = st.empty()
        with thinking_placeholder:
            st.markdown("🍿 **电影摸鱼王正在翻豆瓣、检索 RAG 并思考中...**")

        try:
            # 调用 Agent 获取生成器
            response_generator = st.session_state.agent.chat(prompt)
            
            # 只使用 write_stream，不再额外输出
            full_response = st.write_stream(response_generator)
            
            # 流式完成后清除思考提示
            thinking_placeholder.empty()
            
            # 再次 empty 防止残留
            st.empty()

        except Exception as e:
            thinking_placeholder.empty()
            full_response = f"⚠️ 出错了：{str(e)}"
            st.error(full_response)
            st.empty()

    # 保存完整回答到历史（只保存一次）
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# ====================== 底部按钮 ======================
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = [
            {"role": "assistant", 
             "content": "对话已清空！有什么电影问题尽管问我～ 🍿"}
        ]
        st.rerun()

with col2:
    if st.button("📤 导出对话"):
        history_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" 
                                   for m in st.session_state.messages])
        st.download_button(
            label="下载聊天记录",
            data=history_text,
            file_name="电影摸鱼王_聊天记录.txt",
            mime="text/plain"
        )