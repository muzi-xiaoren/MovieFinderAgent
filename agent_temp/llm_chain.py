from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser  # fallback if needed

response_schemas = [
    ResponseSchema(
        name="answer",
        description="最好详细说明"
    )
]

output_parser = StructuredOutputParser(response_schemas=response_schemas)

# Prompt 模板
template = """你是一个专业的影视资料查询机器，你的任务是回答用户关于影视问题的问题。

用户问题：{question}

{format_instructions}"""

prompt = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

# LLM 配置（你的 GLM-4 兼容 OpenAI 接口）
llm = ChatOpenAI(
    openai_api_key="",
    openai_api_base="https://ark.cn-beijing.volces.com/api/v3",
    model="glm-4-7-251222",
    temperature=0.1,           # 更低 → 更稳定结构化输出
    streaming=False,           # 先关掉 streaming，便于调试解析
)

# ── LCEL 链式调用 ──
# prompt → llm → output_parser
chain = prompt | llm | output_parser   # 最核心的一行

# 执行
question = "查询一下飞驰人生3什么时候上线流媒体"

try:
    print("解析后的结构化结果：")
    for chunk in chain.stream({"question": question}):
        print(chunk.content, end="", flush=True)
    # result
    # result 是一个 dict: {'answer': '内容...'}
except Exception as e:
    print("解析失败：", e)
    # 如果经常失败，可以 fallback 到 StrOutputParser + 手动处理
    fallback_chain = prompt | llm | StrOutputParser()
    raw = fallback_chain.stream({"question": question})
    print("原始回答：")
    print(raw)