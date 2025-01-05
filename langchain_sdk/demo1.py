import os

from dotenv import find_dotenv, load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langgraph.prebuilt import chat_agent_executor

_ = load_dotenv(find_dotenv())
api_key = os.environ['ZHIPUAI_API_KEY']

llm = ChatOpenAI(
    model='glm-4-plus',
    temperature=0.8,
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("你现在是一个Python开发者"),
        MessagesPlaceholder(variable_name='my_msg'),
        # HumanMessagePromptTemplate.from_template('{question}')
    ]
)
search = TavilySearchResults(max_result=2)

agent_exe = chat_agent_executor.create_tool_calling_executor(llm, [search])
chain = prompt | llm  # 如果需要代理，在这儿替换就行
do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg'
)

for resp in do_message.stream(
        {
            'my_msg': [HumanMessage(content="我要在每一行代码后面实时显示git提交记录")]
        },
        config={'configurable': {'session_id': '1212'}}
):
    print(resp.content, end='')
#
# print('/n')
# for resp in do_message.stream(
#         {
#             'my_msg': [HumanMessage(content="讲一讲你对于我的了解")]
#         },
#         config={'configurable': {'session_id': 'libai123'}}
# ):
#     print(resp, end='')
# print('/n')
# for resp in do_message.stream(
#         {
#             'my_msg': [HumanMessage(content="今天我的朋友汪伦要走了，能不能帮我写一首诗饯别")]
#         },
#         config={'configurable': {'session_id': 'libai123'}}
# ):
#     print(resp, end='')
# print('/n')
# for resp in do_message.stream(
#         {
#             'my_msg': [HumanMessage(content="东莞今天的天气怎么样")]
#         },
#         config={'configurable': {'session_id': 'libai123'}}
# ):
#     print(resp, end='')
