# ConversationBufferWindowMemory解释和示例代码用法如下
# ConversationBufferWindowMemory 是一个滑动窗口记忆系统，它会记住最近的 N 条对话记录。
# 示例代码如下：
# from langchain_classic.memory import ConversationBufferWindowMemory
# memory = ConversationBufferWindowMemory(k=5)  # 记住最近 5 条对话
# ConversationBufferWindowMemory 是一种基于窗口的记忆系统，它会记住最近的 N 条对话历史。

# ConversationTokenBufferMemory 解释和示例代码用法如下
# ConversationTokenBufferMemory 是一个基于令牌的记忆系统，它会记住最近的 N 个令牌。
# 示例代码如下：
# from langchain_classic.memory import ConversationTokenBufferMemory
# memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=1000)

# ConversationSummaryMemory 解释和示例代码用法如下
# ConversationSummaryMemory 是一个摘要记忆系统，它会根据对话历史生成一个摘要。
# 示例代码如下：
# from langchain_classic.memory import ConversationSummaryMemory
# memory = ConversationSummaryMemory(llm=llm)


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.chains import ConversationChain
from langchain_core.output_parsers import StrOutputParser

# LLM 配置（使用 GLM-4，兼容 OpenAI 接口）
llm = ChatOpenAI(
    openai_api_key="",
    openai_api_base="https://ark.cn-beijing.volces.com/api/v3",
    model="glm-4-7-251222",
    temperature=0.1,
    streaming=False,           # ConversationChain + 总结通常不建议开启 streaming
)

# 创建带总结的记忆系统
memory = ConversationSummaryMemory(
    llm=llm,
    max_token_limit=500,            # 总结后控制在多少 token 以内，可根据需要调整
    memory_key="history",
    input_key="input",
)

# 结构化输出定义
response_schemas = [
    ResponseSchema(
        name="answer",
        description="对用户影视相关问题的完整、准确回答，最好包含上映时间、平台、剧情简介、评分等详细信息"
    )
]

output_parser = StructuredOutputParser(response_schemas=response_schemas)

# 提示模板（关键：加入 {history}，避免变量冲突）
template = """你是一个专业的影视资料查询机器，你的任务是根据之前的对话历史回答用户关于影视的问题。
请给出准确、详细的回答，不要说废话。

对话历史：
{history}

当前用户问题：{input}

{format_instructions}

请严格按照 format_instructions 的格式输出。"""

# 创建 PromptTemplate，并注入 format_instructions
prompt = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

# 创建 ConversationChain，并覆盖默认的 prompt
conversation_chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,                # 使用我们自定义的带 {history} 和 format_instructions 的 prompt
    verbose=True                  # 打开可以看到总结过程，方便调试，生产环境可改为 False
)

# ───────────────────────────────────────────────
# 主循环：从终端持续读取用户输入
# ───────────────────────────────────────────────

print("影视资料查询机器人已启动！")
print("输入你的问题（输入 exit / quit / bye / 退出 结束）\n")

while True:
    try:
        user_input = input("你: ").strip()
        
        if user_input.lower() in ["exit", "quit", "bye", "退出", "再见"]:
            print("机器人: 感谢使用，再见！")
            break
            
        if not user_input:
            print("机器人: 请输入有效问题哦～")
            continue

        print(f"\n处理中...")

        # 调用 chain（会自动带上历史总结）
        response = conversation_chain.invoke({"input": user_input})
        raw_answer = response["response"]
        
        print("\n机器人原始回答：")
        print(raw_answer)

        # 尝试解析成结构化格式
        try:
            parsed = output_parser.parse(raw_answer)
            print("\n结构化解析结果：")
            print(parsed)
        except Exception as parse_err:
            print("\n结构化解析失败：", parse_err)
            print("（已使用原始回答继续）")

        # 可选：显示当前记忆总结（调试用，可注释掉）
        # print("\n[当前对话总结]")
        # print(memory.load_memory_variables({})["history"])
        # print("-" * 60)

        print("\n" + "-" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n机器人: 检测到 Ctrl+C，已退出。")
        break
    except Exception as e:
        print(f"\n发生错误：{e}")
        print("请再试一次或输入 exit 退出。\n")