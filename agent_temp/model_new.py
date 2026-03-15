from langchain_openai import ChatOpenAI             
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema, PydanticOutputParser   # legacy parser
from pydantic import BaseModel, Field

#定义输出模板
class MovieAnswer(BaseModel):
    answer: str = Field(description="用户问题的答案，直接、完整、准确地回答，不要添加多余解释")

# 创建输出解析器
output_parser = PydanticOutputParser(pydantic_object=MovieAnswer)

# 影视资料查询机器
template = """
你是一个专业的影视资料查询机器，你的任务是回答用户关于影视问题的问题。

用户问题：{question}

{format_instructions}
"""

# 格式化输出（只要求一个字段）
response_schema = ResponseSchema(
    name="answer",
    description="详细的说明情况",
)
prompt = PromptTemplate.from_template(
    template=template,
    partial_variables={
        "format_instructions": output_parser.get_format_instructions(),
    },
)

# 根据prompt生成messages（这里其实是字符串 → 转成 HumanMessage）
question = "你知道美丽人生吗"
formatted_prompt = prompt.format(question=question)

# 因为你用的是 ChatOpenAI，需要包成消息列表
from langchain_core.messages import HumanMessage
messages = [HumanMessage(content=formatted_prompt)]

# 初始化 LLM（你用的实际上是字节的 GLM-4，不是 OpenAI，但用 ChatOpenAI 兼容接口没问题）
llm = ChatOpenAI(
    openai_api_key="",
    openai_api_base="https://ark.cn-beijing.volces.com/api/v3",
    model="glm-4-7-251222",           # GLM-4 系列
    temperature=0.3,                   # 建议稍微低一点，结构化输出更稳定
    streaming=True,
)

# 同步调用
response = llm.invoke(messages)
print("原始回答：")
print(response.content)

# 如果想解析成结构化
try:
    parsed = output_parser.parse(response.content)
    print("\n解析后的结构化结果：")
    print(parsed)
    parsed
    #使用模板Model.py格式化输出

except Exception as e:
    print("解析失败：", e)
    print("原始内容是：", response.content)

    
# 流式输出示例
# for chunk in llm.stream("写一首关于字节跳动的打油诗"):
#     print(chunk.content, end="", flush=True)
