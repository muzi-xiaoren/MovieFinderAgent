# MovieFinderAgent

**一个基于 LangChain 和大语言模型（LLM）的智能电影信息查询 Agent**

通过自然语言与 Agent 对话，即可快速获取电影信息、推荐、评分、演员阵容等内容。目前主要集成豆瓣电影数据源，并支持 RAG 向量增强。

## 项目简介

MovieFinderAgent 使用 LangChain 框架构建，支持用户以自然语言提问，例如：

- “最近五年最好看的科幻电影有哪些？”
- “克里斯托弗·诺兰的所有电影按豆瓣评分排序”
- “《盗梦空间》的演员阵容和评分是多少？”
- “推荐几部类似《寄生虫》的韩国悬疑电影”

Agent 会自动理解意图、调用工具、结合向量检索，并返回结构化的回答。

## 主要功能

- 自然语言意图识别与多轮对话
- 集成电影数据查询工具（当前支持豆瓣）
- RAG（检索增强生成）机制，提升回答准确性
- 可扩展的工具体系（未来可接入 IMDb、TMDB 等更多数据源）
- 命令行交互界面

## 项目结构

```text
MovieFinderAgent/
├── agent_temp/          # Agent 运行过程中的临时文件
├── apis/                # 外部工具实现
│   └── douban_tool.py   # 豆瓣电影查询工具
├── logs            	 # 每次对话产生的详细日志内容
├── rag/                 # RAG 相关模块（向量库、嵌入、检索）
├── config.py            # 配置项：LLM 参数、API Key、工具开关等
├── main.py              # 程序入口（命令行交互）
├── movieFinder.py       # Agent 核心逻辑（工具绑定、执行流程）
├── LICENSE              # Apache-2.0 许可证
└── README.md
```



## 技术栈

- **语言**：Python
- **核心框架**：LangChain
- **组件**：LLM（兼容 OpenAI 接口的模型）、RAG、工具调用（Tool Calling）
- **数据源**：豆瓣电影（通过 douban_tool.py 实现）

## 安装与使用

### 1. 克隆仓库

Bash

```
git clone https://github.com/muzi-xiaoren/MovieFinderAgent.git cd MovieFinderAgent
```

### 2. 安装依赖

Bash

```
pip install langchain langchain-community langchain-huggingface faiss-cpu loguru # 如需使用特定 LLM，可额外安装对应 SDK
```

### 3. 配置 LLM

编辑 config.py 文件，填入你的 LLM 配置（当前支持兼容 OpenAI 接口的模型，例如火山引擎 GLM-4、OpenAI 等）：

```
# 示例配置
LLM_CONFIG = {
    "api_key": "your_api_key_here",
    "base_url": "https://your-llm-endpoint.com/v1",  # 如使用火山引擎等
    "model": "glm-4",                               # 或 "gpt-4o" 等
    "temperature": 0.3,
}
```

### 4. 初始化 RAG 向量库（首次运行建议执行）

根据 rag/ 目录下的脚本构建或更新向量库（具体命令请参考仓库内相关文件）。

### 5. 启动 Agent

```
python main.py
```

启动后，在命令行输入电影相关问题即可。输入 exit 或 退出 结束会话。

## 使用示例

**输入：**

```
最近五年最好看的科幻电影有哪些？
```

**Agent 将自动调用工具并返回：**

- 电影列表
- 豆瓣评分
- 相关信息（导演、年份等）

支持多轮对话，例如后续追问“其中哪一部是诺兰执导的？”。

## 开发与扩展

- **添加新工具**：在 apis/ 目录下新增工具文件，并在 movieFinder.py 中绑定。
- **RAG 增强**：rag/ 模块负责向量存储与检索，可根据需要调整嵌入模型或知识库内容。
- **配置调整**：所有核心参数集中在 config.py 中。