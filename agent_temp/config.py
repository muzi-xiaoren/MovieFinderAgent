"""配置文件"""

# LLM 配置
LLM_CONFIG = {
    "openai_api_key": "bd428d1e-5e82-490c-9412-f80ada257612",
    "openai_api_base": "",
    "model": "glm-4-7-251222",
    "temperature": 0.3,
    "streaming": True,
    "stream_usage": False,
}

# Agent 系统提示词
SYSTEM_PROMPT = """你是一个专业的影视资料查询助手。

你可以使用以下工具帮助用户：
1. search_douban_movies: 搜索豆瓣电影信息，获取详细资料

工作流程：
- 如果用户询问具体电影信息，先调用工具搜索最新资料
- 结合向量数据库中的豆瓣Top250信息和搜索结果回答
- 用自然、流畅的中文回答，信息要详细专业

回答要求：
- 基于搜索结果和已有知识回答
- 可以用标题、编号、分段让内容更清晰
- 不要出现 ```json
- 不要输出多余说明，直接开始回答正文"""

# 会话配置
SESSION_ID = "movie_finder_user"

# 向量检索配置
RETRIEVAL_K = 1