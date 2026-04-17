import streamlit as st
import sys
from pathlib import Path
import random
from movieFinder import MovieFinder
import asyncio

# Windows + Playwright + Streamlit 兼容性修复
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

# ====================== 初始化 Agent（只加载一次） ======================
if "agent" not in st.session_state:
    with st.spinner("正在初始化电影摸鱼王 Agent... 这可能需要几秒钟"):
        try:
            st.session_state.agent = MovieFinder()
            st.success("✅ Agent 初始化完成！")
        except Exception as e:
            st.error(f"初始化失败: {e}")
            st.stop()

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

# ====================== 用户输入 ======================
if prompt := st.chat_input("输入你的电影问题，例如：最近好看的悬疑片？"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # 显示助手回复（带等待提示 + 流式输出）
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # 先显示等待提示
        with message_placeholder:
            st.markdown("🍿 **电影摸鱼王正在翻豆瓣、检索 RAG 并思考中...**")
            with st.spinner(""):
                pass  # 只显示 spinner
        try:
            # 调用 chat，获取 generator
            response_generator = st.session_state.agent.chat(prompt)
            # 使用真正的流式输出（打字机效果）
            full_response = st.write_stream(response_generator)
        except Exception as e:
            full_response = f"⚠️ 出错了：{str(e)}"
            st.error(full_response)

    # 保存完整回答到历史
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# ====================== 底部按钮 ======================
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = [
            {"role": "assistant", "content": "对话已清空！有什么电影问题尽管问我～ 🍿"}
        ]
        st.rerun()

with col2:
    if st.button("📤 导出对话"):
        history_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button(
            label="下载聊天记录",
            data=history_text,
            file_name="电影摸鱼王_聊天记录.txt",
            mime="text/plain"
        )