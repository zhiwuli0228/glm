from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv, find_dotenv
import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import chat_agent_executor

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

HOST = '47.119.150.113'
POET = 3306
USER = 'root'
PASSWORD = 'root'
TABLESPACE = 'user'

mysql_url = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USER, PASSWORD, HOST, POET, TABLESPACE)

db = SQLDatabase.from_uri(mysql_url)

# 数据库执行工具类
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# 构建prompt
system_prompt = """
您是一个被设计用来与SQL数据库交互的代理
给定一个输入问题，创建一个语法正确的SQL语句并执行，然后查看查询结果并返回答案。
除非用户指定了他们想要获得的示例的数量，否则始终将SQL查询限制为最多10个结果。
你可以按相关列对结果进行排序，以返回MySql数据库中最匹配的数据
您可以使用语数据库交互的工具。在执行查询之前，你必须仔细检查。如果在执行查询时出现错误，请重写查询并重试
不要对数据库做任何DML语句（插入，更新，删除等）

首先，你应该查询数据库中的表，看看可以查询什么
不可以跳过这一步
然后查询最相关的表的模式
"""

system_message = SystemMessage(content=system_prompt)

agent_executor = chat_agent_executor.create_tool_calling_executor(model=llm, tools=tools,
                                                                  state_modifier=system_message)
resp = agent_executor.invoke({'messages': [HumanMessage(content="年龄大于20岁的学生有哪几个？")]})
result = resp['messages']
print(result[len(result) - 1])
