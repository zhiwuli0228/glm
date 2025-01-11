import os
import bs4
from dotenv import find_dotenv, load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

from glm.zhipuai_embedding import ZhipuAIEmbeddings

_ = load_dotenv(find_dotenv())
os.environ[
    'USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
api_key = os.environ['ZHIPUAI_API_KEY']

llm = ChatOpenAI(
    model='glm-4-plus',
    temperature=0.8,
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 加载网页数据
loader = WebBaseLoader(
    web_path='https://lilianweng.github.io/posts/2023-06-23-agent/',
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=('post-header', 'post-title', 'post-content'))
    )
)

docs = loader.load()

# print(docs)
# 2. 大文本切割
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10
)
splits = splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=ZhipuAIEmbeddings())

# 3.检索器
retriever = vectorstore.as_retriever()

system_prompt = """
请根据下面检索到的资料回答问题，如果不知道请回答不知道，不要编答案，如果有正确答案，尽量使用简洁的语言进行表达
{context}
"""

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        HumanMessagePromptTemplate.from_template('{input}')
    ]
)

chain1 = create_stuff_documents_chain(llm, prompt)

# chain2 = create_retrieval_chain(retriever, chain1)

# resp = chain2.invoke({'input': '什么是任务拆解'})
# print(resp['answer'])

# 构建子链的提示模版
contextualize_q_system_prompt = """通过聊天上下文生成一个可以被理解的历史聊天记录和用户最近的提问，
你不需要回答，只需要转述或者直接返回生产的内容
"""

retriever_history_temp = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        HumanMessagePromptTemplate.from_template('{input}')
    ]
)

history_chain = create_history_aware_retriever(llm, retriever, retriever_history_temp)

# 构建历史记录查询
store = {}


def get_history_by_session_id(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


chain = create_retrieval_chain(history_chain, chain1)

do_message = RunnableWithMessageHistory(
    chain,
    get_history_by_session_id,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)


